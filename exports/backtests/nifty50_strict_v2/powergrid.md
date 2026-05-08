# POWERGRID (POWERGRID)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 313.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 9 |
| PENDING | 33 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 16 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 17
- **Target hits / Stop hits / Partials:** 3 / 21 / 4
- **Avg / median % per leg:** 0.49% / -1.62%
- **Sum % (uncompounded):** 13.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 2 | 11 | 1 | 0.03% | 0.4% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.26% | -7.6% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.99% | 8.0% |
| SELL (all) | 14 | 7 | 50.0% | 1 | 10 | 3 | 0.95% | 13.3% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.36% | 20.1% |
| SELL @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 1 | 7 | 0 | -0.85% | -6.8% |
| retest1 (combined) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.05% | 12.5% |
| retest2 (combined) | 16 | 3 | 18.8% | 3 | 13 | 0 | 0.07% | 1.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 11:15:00 | 190.05 | 186.30 | 186.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 191.14 | 186.41 | 186.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 194.65 | 195.00 | 192.01 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-09 13:15:00 | 195.90 | 195.02 | 192.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 14:15:00 | 196.00 | 195.03 | 192.10 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 09:15:00 | 205.80 | 196.69 | 193.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-10-25 09:15:00 | 198.75 | 198.87 | 195.20 | SL hit (close<ema200) qty=0.50 sl=198.87 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 264.25 | 277.58 | 263.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-14 10:15:00 | 265.75 | 277.19 | 263.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 11:15:00 | 266.95 | 277.09 | 263.41 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-15 11:15:00 | 261.80 | 276.28 | 263.47 | SL hit (close<static) qty=1.00 sl=261.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-15 14:15:00 | 265.85 | 275.86 | 263.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-15 15:15:00 | 265.50 | 275.76 | 263.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-21 09:15:00 | 271.10 | 273.19 | 263.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 271.10 | 273.16 | 263.42 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2024-04-30 10:15:00 | 298.21 | 281.61 | 274.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 318.60 | 335.51 | 335.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 313.85 | 332.12 | 333.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-12 13:15:00 | 322.90 | 325.42 | 329.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 322.45 | 325.39 | 329.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 330.60 | 322.95 | 327.23 | SL hit (close>static) qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-04 11:15:00 | 322.55 | 327.46 | 328.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 322.50 | 327.41 | 328.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 319.95 | 327.25 | 328.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 322.30 | 327.21 | 328.65 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 330.60 | 327.24 | 328.62 | SL hit (close>static) qty=1.00 sl=329.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 330.60 | 327.24 | 328.62 | SL hit (close>static) qty=1.00 sl=329.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-18 09:15:00 | 323.30 | 328.39 | 328.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 322.45 | 328.33 | 328.89 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Target hit | 2025-01-13 13:15:00 | 290.21 | 314.33 | 320.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 278.40 | 269.70 | 280.58 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 305.00 | 286.08 | 286.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.67 | 286.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.90 | 300.84 | 295.36 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 308.10 | 300.86 | 295.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:15:00 | 308.00 | 300.93 | 295.62 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.17 | 296.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-14 12:15:00 | 294.90 | 301.11 | 296.13 | SL hit (close<ema400) qty=1.00 sl=296.13 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 299.30 | 300.61 | 296.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 299.45 | 300.60 | 296.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-21 13:15:00 | 298.05 | 300.60 | 296.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-21 14:15:00 | 296.05 | 300.56 | 296.65 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 289.35 | 300.40 | 296.61 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 298.10 | 299.70 | 296.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 298.05 | 299.68 | 296.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 293.25 | 299.53 | 296.52 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 12:15:00 | 299.30 | 296.25 | 295.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:15:00 | 300.05 | 296.29 | 295.47 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 295.10 | 296.71 | 295.74 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.15 | 294.84 | 294.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 299.10 | 294.42 | 294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 301.15 | 294.68 | 294.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-21 11:15:00 | 296.30 | 295.77 | 295.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:15:00 | 297.00 | 295.78 | 295.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 11:15:00 | 297.10 | 295.83 | 295.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 297.75 | 295.85 | 295.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.17 | 295.50 | SL hit (close<static) qty=1.00 sl=294.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.17 | 295.50 | SL hit (close<static) qty=1.00 sl=294.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 290.30 | 290.95 | 292.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 15:15:00 | 290.35 | 290.94 | 292.57 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 275.83 | 288.62 | 291.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 287.15 | 286.52 | 289.64 | SL hit (close>ema200) qty=0.50 sl=286.52 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-12 12:15:00 | 285.70 | 285.80 | 288.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 13:15:00 | 286.85 | 285.81 | 288.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 284.10 | 286.95 | 288.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 284.40 | 286.92 | 288.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 289.45 | 285.65 | 287.41 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 285.15 | 285.68 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 285.55 | 285.68 | 287.40 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 289.40 | 285.75 | 287.38 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 285.55 | 285.98 | 287.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 11:15:00 | 285.80 | 285.97 | 287.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 13:15:00 | 284.70 | 285.96 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 285.20 | 285.95 | 287.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 285.40 | 285.91 | 287.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 14:15:00 | 286.05 | 285.91 | 287.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.33 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 285.55 | 286.15 | 287.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-13 14:15:00 | 285.75 | 286.14 | 287.35 | ENTRY2 sustain failed after 60m |

### Cycle 7 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.16 | 288.15 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.16 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.81 | 288.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.70 | 268.51 | 274.00 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 15:15:00 | 267.00 | 268.48 | 273.88 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 267.50 | 268.46 | 273.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 13:15:00 | 266.75 | 268.44 | 273.54 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.19 | 272.08 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 267.40 | 267.63 | 271.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 268.05 | 267.63 | 271.95 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.65 | 263.49 | 268.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.41 | 263.49 | 268.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.23 | 266.19 | SL hit (close>ema200) qty=0.50 sl=261.23 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.23 | 266.19 | SL hit (close>ema200) qty=0.50 sl=261.23 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 279.70 | 261.16 | 265.79 | SL hit (close>static) qty=1.00 sl=272.10 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.76 | 269.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.85 | 270.22 | 269.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.80 | 290.27 | 282.98 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.31 | 283.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.90 | 290.33 | 283.11 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.05 | 290.36 | 283.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.41 | 283.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.05 | 293.05 | 285.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.09 | 285.77 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.29 | 288.79 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.30 | 288.83 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.28 | 289.35 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | SL hit (close<ema400) qty=1.00 sl=289.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | SL hit (close<ema400) qty=1.00 sl=289.33 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | SL hit (close<ema400) qty=1.00 sl=289.33 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 291.75 | 294.72 | 289.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-06 10:15:00 | 287.95 | 294.65 | 289.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.55 | 289.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.54 | 289.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-27 10:15:00 | 322.80 | 304.58 | 296.95 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-09 14:15:00 | 196.00 | 2023-10-17 09:15:00 | 205.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-10-09 14:15:00 | 196.00 | 2023-10-25 09:15:00 | 198.75 | STOP_HIT | 0.50 | 1.40% |
| BUY | retest2 | 2024-03-14 11:15:00 | 266.95 | 2024-03-15 11:15:00 | 261.80 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-03-21 10:15:00 | 271.10 | 2024-04-30 10:15:00 | 298.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-12 14:15:00 | 322.45 | 2024-11-22 09:15:00 | 330.60 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-12-04 12:15:00 | 322.50 | 2024-12-06 09:15:00 | 330.60 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-12-05 10:15:00 | 322.30 | 2024-12-06 09:15:00 | 330.60 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-12-18 10:15:00 | 322.45 | 2025-01-13 13:15:00 | 290.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-05-12 10:15:00 | 308.00 | 2025-05-14 12:15:00 | 294.90 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-05-15 14:15:00 | 299.45 | 2025-05-22 09:15:00 | 289.35 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-23 12:15:00 | 298.05 | 2025-05-27 09:15:00 | 293.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-09 13:15:00 | 300.05 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-22 12:15:00 | 297.75 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2025-08-18 15:15:00 | 290.35 | 2025-08-28 09:15:00 | 275.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-18 15:15:00 | 290.35 | 2025-09-02 10:15:00 | 287.15 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2025-09-25 15:15:00 | 284.40 | 2025-10-03 14:15:00 | 289.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-06 10:15:00 | 285.55 | 2025-10-07 09:15:00 | 289.40 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-08 14:15:00 | 285.20 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-20 14:15:00 | 253.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.75 | 2026-01-20 14:15:00 | 253.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.75 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-02-03 09:15:00 | 279.70 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-04-06 13:15:00 | 293.45 | 2026-04-27 10:15:00 | 322.80 | TARGET_HIT | 1.00 | 10.00% |
