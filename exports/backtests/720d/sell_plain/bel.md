# BEL (BEL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 440.60
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
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 20 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 1 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -3.37% / -2.94%
- **Sum % (uncompounded):** -43.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 1 | 7.7% | 0 | 13 | 0 | -3.37% | -43.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.98% | -2.0% |
| SELL @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 0 | 12 | 0 | -3.49% | -41.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.98% | -2.0% |
| retest2 (combined) | 12 | 1 | 8.3% | 0 | 12 | 0 | -3.49% | -41.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 284.15 | 299.12 | 299.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 275.05 | 294.41 | 296.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 292.15 | 291.36 | 294.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 289.50 | 291.36 | 294.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 289.50 | 291.36 | 294.57 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-25 11:15:00 | 289.30 | 291.32 | 294.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-25 13:15:00 | 289.70 | 291.28 | 294.47 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-09-30 09:15:00 | 286.20 | 291.16 | 294.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 11:15:00 | 287.20 | 291.08 | 294.08 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-30 13:15:00 | 287.80 | 281.85 | 286.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 15:15:00 | 287.95 | 281.98 | 286.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 295.65 | 282.66 | 286.35 | SL hit (close>static) qty=1.00 sl=295.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 295.65 | 282.66 | 286.35 | SL hit (close>static) qty=1.00 sl=295.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-12 15:15:00 | 289.25 | 287.07 | 288.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 280.25 | 287.01 | 288.15 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 297.95 | 284.89 | 286.76 | SL hit (close>static) qty=1.00 sl=295.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-30 13:15:00 | 287.55 | 299.84 | 296.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 15:15:00 | 286.20 | 299.55 | 296.85 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 296.45 | 298.37 | 296.47 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 296.45 | 298.37 | 296.47 | SL hit (close>static) qty=1.00 sl=295.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-03 14:15:00 | 292.00 | 298.10 | 296.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 289.45 | 297.96 | 296.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 283.25 | 294.91 | 294.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 283.25 | 294.91 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 281.30 | 294.66 | 294.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 286.90 | 286.88 | 290.35 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-20 15:15:00 | 285.55 | 286.85 | 290.29 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:15:00 | 282.60 | 286.81 | 290.25 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 288.20 | 279.51 | 285.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 288.20 | 279.51 | 285.28 | SL hit (close>ema400) qty=1.00 sl=285.28 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-01 12:15:00 | 273.35 | 280.61 | 285.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-01 13:15:00 | 281.60 | 280.62 | 285.54 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-01 15:15:00 | 280.65 | 280.63 | 285.50 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 267.80 | 280.50 | 285.41 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 2520m) |
| Stop hit — per-position SL triggered | 2025-02-05 10:15:00 | 290.55 | 280.33 | 284.96 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-06 13:15:00 | 280.35 | 280.95 | 285.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 280.00 | 280.93 | 285.00 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-13 13:15:00 | 281.35 | 268.51 | 273.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 280.10 | 268.74 | 273.24 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 290.35 | 270.91 | 274.00 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 290.35 | 270.91 | 274.00 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 280.40 | 283.55 | 280.48 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 281.05 | 283.52 | 280.48 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 266.90 | 283.36 | 280.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 268.15 | 283.08 | 280.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-08 10:15:00 | 279.25 | 282.55 | 280.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-08 12:15:00 | 281.30 | 282.50 | 280.35 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2025-04-08 15:15:00 | 282.10 | 282.47 | 280.37 | SL hit (close>static) qty=1.00 sl=281.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-09 09:15:00 | 277.95 | 282.42 | 280.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-09 11:15:00 | 281.00 | 282.38 | 280.35 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 293.00 | 282.63 | 280.60 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.44 | 381.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.90 | 405.68 | 405.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.90 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 400.55 | 400.43 | 402.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:15:00 | 399.55 | 400.42 | 402.65 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 404.35 | 400.46 | 402.66 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 400.85 | 400.54 | 402.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 398.55 | 400.50 | 402.63 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 404.85 | 399.50 | 401.79 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-21 10:15:00 | 400.05 | 407.49 | 405.85 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-21 11:15:00 | 402.90 | 407.45 | 405.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-21 13:15:00 | 400.40 | 407.36 | 405.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-21 14:15:00 | 402.35 | 407.31 | 405.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-30 14:15:00 | 400.70 | 431.95 | 429.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-03-30 15:15:00 | 401.60 | 431.65 | 429.67 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-30 11:15:00 | 287.20 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-10-30 15:15:00 | 287.95 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-11-13 09:15:00 | 280.25 | 2024-11-25 10:15:00 | 297.95 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2024-12-30 15:15:00 | 286.20 | 2025-01-02 13:15:00 | 296.45 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-01-06 09:15:00 | 289.45 | 2025-01-09 12:15:00 | 283.25 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest1 | 2025-01-21 09:15:00 | 282.60 | 2025-01-31 09:15:00 | 288.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-02-03 09:15:00 | 267.80 | 2025-02-05 10:15:00 | 290.55 | STOP_HIT | 1.00 | -8.50% |
| SELL | retest2 | 2025-02-06 15:15:00 | 280.00 | 2025-03-19 11:15:00 | 290.35 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-03-13 15:15:00 | 280.10 | 2025-03-19 11:15:00 | 290.35 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2025-04-02 11:15:00 | 280.40 | 2025-04-08 15:15:00 | 282.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-04-07 11:15:00 | 268.15 | 2025-04-15 09:15:00 | 293.00 | STOP_HIT | 1.00 | -9.27% |
| SELL | retest2 | 2025-12-24 15:15:00 | 399.55 | 2025-12-26 09:15:00 | 404.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-12-26 14:15:00 | 398.55 | 2026-01-02 09:15:00 | 404.85 | STOP_HIT | 1.00 | -1.58% |
