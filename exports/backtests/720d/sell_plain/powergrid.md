# POWERGRID (POWERGRID)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 314.00
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
| ALERT3 | 4 |
| PENDING | 22 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 2 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** 1.20% / -0.57%
- **Sum % (uncompounded):** 15.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 0 | 12 | 1 | 1.20% | 15.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.42% | -0.8% |
| SELL @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 0 | 10 | 1 | 1.50% | 16.5% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.42% | -0.8% |
| retest2 (combined) | 11 | 5 | 45.5% | 0 | 10 | 1 | 1.50% | 16.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 331.50 | 337.17 | 337.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 329.90 | 336.62 | 336.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-12 13:15:00 | 322.90 | 325.33 | 329.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:15:00 | 322.65 | 325.28 | 329.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 330.60 | 322.89 | 327.54 | SL hit (close>static) qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-04 11:15:00 | 322.60 | 327.42 | 329.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 13:15:00 | 324.05 | 327.34 | 328.99 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 320.00 | 327.23 | 328.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-05 11:15:00 | 325.40 | 327.16 | 328.86 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 330.55 | 327.22 | 328.84 | SL hit (close>static) qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-18 09:15:00 | 323.25 | 328.37 | 329.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:15:00 | 323.20 | 328.26 | 329.02 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-04 09:15:00 | 274.72 | 300.51 | 308.98 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 272.85 | 269.43 | 283.37 | SL hit (close>ema200) qty=0.50 sl=269.43 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.10 | 294.84 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-25 15:15:00 | 291.85 | 296.07 | 295.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-28 09:15:00 | 292.85 | 296.04 | 295.43 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-07-28 14:15:00 | 291.90 | 295.91 | 295.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-28 15:15:00 | 292.30 | 295.87 | 295.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-29 09:15:00 | 291.25 | 295.83 | 295.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-29 11:15:00 | 293.05 | 295.75 | 295.31 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-07-30 14:15:00 | 289.35 | 295.56 | 295.24 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 291.35 | 295.47 | 295.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 291.75 | 295.38 | 295.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 291.05 | 295.30 | 295.11 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-01 14:15:00 | 290.95 | 294.96 | 294.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 290.30 | 290.96 | 292.57 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-18 15:15:00 | 290.70 | 290.95 | 292.57 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 289.50 | 290.94 | 292.55 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 287.80 | 290.89 | 292.51 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 288.60 | 285.86 | 288.43 | SL hit (close>ema400) qty=1.00 sl=288.43 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 284.00 | 286.95 | 288.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 281.25 | 286.86 | 288.29 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 289.45 | 285.66 | 287.41 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 285.15 | 285.68 | 287.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 285.35 | 285.67 | 287.39 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 289.55 | 285.75 | 287.38 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 285.55 | 285.97 | 287.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-08 11:15:00 | 285.80 | 285.97 | 287.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 13:15:00 | 284.70 | 285.96 | 287.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 285.00 | 285.94 | 287.38 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 285.40 | 285.91 | 287.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-09 14:15:00 | 286.05 | 285.91 | 287.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.33 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 285.60 | 286.15 | 287.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-13 14:15:00 | 285.75 | 286.15 | 287.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-04 09:15:00 | 280.60 | 288.52 | 288.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 280.05 | 288.35 | 288.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-24 09:15:00 | 267.90 | 268.47 | 273.84 | ENTRY1 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 267.50 | 268.45 | 273.57 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 14:15:00 | 265.40 | 268.41 | 273.50 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 267.40 | 267.61 | 271.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 267.60 | 267.62 | 271.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 266.90 | 260.37 | 265.29 | SL hit (close>ema400) qty=1.00 sl=265.29 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 279.70 | 260.75 | 265.40 | SL hit (close>static) qty=1.00 sl=272.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 15:15:00 | 322.65 | 2024-11-22 09:15:00 | 330.60 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-12-04 13:15:00 | 324.05 | 2024-12-06 09:15:00 | 330.55 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-12-18 11:15:00 | 323.20 | 2025-02-04 09:15:00 | 274.72 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-18 11:15:00 | 323.20 | 2025-03-10 09:15:00 | 272.85 | STOP_HIT | 0.50 | 15.58% |
| SELL | retest2 | 2025-07-31 09:15:00 | 291.35 | 2025-08-01 15:15:00 | 290.40 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-07-31 14:15:00 | 291.05 | 2025-08-01 15:15:00 | 290.40 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest1 | 2025-08-19 11:15:00 | 287.80 | 2025-09-12 15:15:00 | 288.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-09-26 09:15:00 | 281.25 | 2025-10-03 14:15:00 | 289.45 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-10-06 11:15:00 | 285.35 | 2025-10-07 09:15:00 | 289.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-10-08 15:15:00 | 285.00 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-04 11:15:00 | 280.05 | 2025-11-04 13:15:00 | 278.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest1 | 2025-12-26 14:15:00 | 265.40 | 2026-02-02 13:15:00 | 266.90 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-01-06 13:15:00 | 267.60 | 2026-02-03 09:15:00 | 279.70 | STOP_HIT | 1.00 | -4.52% |
