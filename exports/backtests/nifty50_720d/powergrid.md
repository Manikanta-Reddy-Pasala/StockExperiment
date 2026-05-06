# POWERGRID (POWERGRID.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 315.95
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 8 |
| PENDING | 29 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 7 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 18
- **Target hits / Stop hits / Partials:** 0 / 20 / 1
- **Avg / median % per leg:** -0.57% / -1.51%
- **Sum % (uncompounded):** -11.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.86% | -16.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.72% | -10.9% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.17% | -5.9% |
| SELL (all) | 12 | 3 | 25.0% | 0 | 11 | 1 | 0.40% | 4.8% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.09% | -3.3% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 0 | 8 | 1 | 0.90% | 8.1% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -2.02% | -14.1% |
| retest2 (combined) | 14 | 2 | 14.3% | 0 | 13 | 1 | 0.16% | 2.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 318.60 | 335.51 | 335.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 313.85 | 332.12 | 333.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-12 13:15:00 | 322.90 | 325.42 | 329.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 322.45 | 325.39 | 329.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 329.90 | 322.95 | 327.23 | SL hit qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-04 11:15:00 | 322.55 | 327.46 | 328.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 322.50 | 327.41 | 328.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 319.95 | 327.25 | 328.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 322.30 | 327.21 | 328.65 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-05 14:15:00 | 329.90 | 327.20 | 328.61 | SL hit qty=1.00 sl=329.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-05 14:15:00 | 329.90 | 327.20 | 328.61 | SL hit qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-18 09:15:00 | 323.30 | 328.39 | 328.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 322.45 | 328.33 | 328.89 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-04 09:15:00 | 274.08 | 300.95 | 309.42 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 278.40 | 269.70 | 280.58 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 305.00 | 286.08 | 286.00 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 305.00 | 286.08 | 286.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.67 | 286.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.90 | 300.84 | 295.36 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 308.10 | 300.86 | 295.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:15:00 | 308.00 | 300.93 | 295.62 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.17 | 296.13 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 296.13 | 301.17 | 296.13 | SL hit qty=1.00 sl=296.13 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 299.30 | 300.61 | 296.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 299.45 | 300.60 | 296.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-21 13:15:00 | 298.05 | 300.60 | 296.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-21 14:15:00 | 296.05 | 300.56 | 296.65 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 295.60 | 300.40 | 296.61 | SL hit qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 298.10 | 299.70 | 296.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 298.05 | 299.68 | 296.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 295.60 | 299.53 | 296.52 | SL hit qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 12:15:00 | 299.30 | 296.25 | 295.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:15:00 | 300.05 | 296.29 | 295.47 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 295.60 | 296.71 | 295.74 | SL hit qty=1.00 sl=295.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.15 | 294.84 | 294.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 299.10 | 294.42 | 294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 301.15 | 294.68 | 294.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 11:15:00 | 296.30 | 295.77 | 295.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:15:00 | 297.00 | 295.78 | 295.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 11:15:00 | 297.10 | 295.83 | 295.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 297.75 | 295.85 | 295.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 294.00 | 296.25 | 295.53 | SL hit qty=1.00 sl=294.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 294.00 | 296.25 | 295.53 | SL hit qty=1.00 sl=294.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 290.30 | 290.95 | 292.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 15:15:00 | 290.35 | 290.94 | 292.57 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 288.51 | 285.75 | 288.51 | SL hit qty=1.00 sl=288.51 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-12 12:15:00 | 285.70 | 285.80 | 288.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 13:15:00 | 286.85 | 285.81 | 288.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 284.10 | 286.95 | 288.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 284.40 | 286.92 | 288.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 288.70 | 285.65 | 287.41 | SL hit qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 285.15 | 285.68 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 285.55 | 285.68 | 287.40 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 288.70 | 285.75 | 287.38 | SL hit qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 285.55 | 285.98 | 287.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 11:15:00 | 285.80 | 285.97 | 287.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 13:15:00 | 284.70 | 285.96 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 285.20 | 285.95 | 287.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 285.40 | 285.91 | 287.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 14:15:00 | 286.05 | 285.91 | 287.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 288.70 | 285.96 | 287.33 | SL hit qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 285.55 | 286.15 | 287.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-13 14:15:00 | 285.75 | 286.14 | 287.35 | ENTRY2 sustain failed after 60m |

### Cycle 6 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.16 | 288.15 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.16 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.81 | 288.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.70 | 268.51 | 274.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 15:15:00 | 267.00 | 268.48 | 273.88 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 267.50 | 268.46 | 273.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 13:15:00 | 266.75 | 268.44 | 273.54 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.19 | 272.08 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 272.08 | 267.19 | 272.08 | SL hit qty=1.00 sl=272.08 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 272.08 | 267.19 | 272.08 | SL hit qty=1.00 sl=272.08 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 267.40 | 267.63 | 271.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 268.05 | 267.63 | 271.95 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 272.10 | 261.16 | 265.79 | SL hit qty=1.00 sl=272.10 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.76 | 269.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.85 | 270.22 | 269.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.80 | 290.27 | 282.98 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.31 | 283.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.90 | 290.33 | 283.11 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.05 | 290.36 | 283.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.41 | 283.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.05 | 293.05 | 285.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.09 | 285.77 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.29 | 288.79 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.30 | 288.83 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.28 | 289.35 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 289.33 | 295.19 | 289.33 | SL hit qty=1.00 sl=289.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 289.33 | 295.19 | 289.33 | SL hit qty=1.00 sl=289.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 289.33 | 295.19 | 289.33 | SL hit qty=1.00 sl=289.33 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 291.75 | 294.72 | 289.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-06 10:15:00 | 287.95 | 294.65 | 289.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.55 | 289.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.54 | 289.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 14:15:00 | 322.45 | 2024-11-22 09:15:00 | 329.90 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-12-04 12:15:00 | 322.50 | 2024-12-05 14:15:00 | 329.90 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-12-05 10:15:00 | 322.30 | 2024-12-05 14:15:00 | 329.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-18 10:15:00 | 322.45 | 2025-02-04 09:15:00 | 274.08 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-18 10:15:00 | 322.45 | 2025-04-16 10:15:00 | 305.00 | STOP_HIT | 0.50 | 5.41% |
| BUY | retest1 | 2025-05-12 10:15:00 | 308.00 | 2025-05-14 11:15:00 | 296.13 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-05-15 14:15:00 | 299.45 | 2025-05-22 09:15:00 | 295.60 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-05-23 12:15:00 | 298.05 | 2025-05-27 09:15:00 | 295.60 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-09 13:15:00 | 300.05 | 2025-06-11 13:15:00 | 295.60 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 10:15:00 | 294.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-22 12:15:00 | 297.75 | 2025-07-25 10:15:00 | 294.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest1 | 2025-08-18 15:15:00 | 290.35 | 2025-09-11 12:15:00 | 288.51 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-09-25 15:15:00 | 284.40 | 2025-10-03 14:15:00 | 288.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-06 10:15:00 | 285.55 | 2025-10-07 09:15:00 | 288.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-08 14:15:00 | 285.20 | 2025-10-10 09:15:00 | 288.70 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-02 11:15:00 | 272.08 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.75 | 2026-01-02 11:15:00 | 272.08 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-02-03 09:15:00 | 272.10 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 289.33 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 289.33 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 289.33 | STOP_HIT | 1.00 | -2.34% |
