# BEL (BEL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 439.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 10 |
| PENDING | 33 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 5 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 1
- **Avg / median % per leg:** -0.13% / -1.44%
- **Sum % (uncompounded):** -2.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.07% | -10.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.68% | -6.7% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | -0.67% | -4.0% |
| SELL (all) | 13 | 4 | 30.8% | 3 | 9 | 1 | 0.60% | 7.8% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 2 | 9 | 0 | -0.66% | -7.2% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.38% | 8.3% |
| retest2 (combined) | 17 | 3 | 17.6% | 3 | 14 | 0 | -0.66% | -11.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 282.65 | 294.51 | 294.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 274.95 | 294.32 | 294.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 292.15 | 291.29 | 292.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 15:15:00 | 292.95 | 291.31 | 292.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 292.95 | 291.31 | 292.78 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-09-25 09:15:00 | 289.50 | 291.29 | 292.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:15:00 | 289.80 | 291.27 | 292.75 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-27 11:15:00 | 290.85 | 291.13 | 292.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 290.20 | 291.12 | 292.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 293.75 | 291.15 | 292.56 | SL hit (close>static) qty=1.00 sl=293.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 293.75 | 291.15 | 292.56 | SL hit (close>static) qty=1.00 sl=293.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-30 09:15:00 | 286.20 | 291.11 | 292.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:15:00 | 286.20 | 291.06 | 292.49 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-10-25 10:15:00 | 257.58 | 283.20 | 286.92 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-30 11:15:00 | 289.80 | 281.68 | 285.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-30 12:15:00 | 291.80 | 281.78 | 285.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-30 13:15:00 | 287.75 | 281.84 | 285.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 14:15:00 | 288.70 | 281.91 | 285.76 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 287.20 | 282.05 | 285.77 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-31 13:15:00 | 283.90 | 282.15 | 285.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-31 14:15:00 | 285.00 | 282.18 | 285.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-04 09:15:00 | 284.10 | 282.22 | 285.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 281.45 | 282.22 | 285.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 292.10 | 282.42 | 285.61 | SL hit (close>static) qty=1.00 sl=287.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 295.65 | 282.56 | 285.66 | SL hit (close>static) qty=1.00 sl=293.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-13 09:15:00 | 280.30 | 286.95 | 287.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:15:00 | 284.20 | 286.92 | 287.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 294.75 | 284.73 | 286.25 | SL hit (close>static) qty=1.00 sl=287.90 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 308.60 | 287.76 | 287.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 310.00 | 291.72 | 289.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 303.10 | 303.45 | 297.40 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-18 15:15:00 | 304.50 | 303.45 | 297.52 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 297.65 | 303.39 | 297.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 297.65 | 303.39 | 297.52 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-19 11:15:00 | 300.40 | 303.31 | 297.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-19 12:15:00 | 299.40 | 303.28 | 297.55 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-20 10:15:00 | 300.65 | 303.06 | 297.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 300.35 | 303.03 | 297.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 293.35 | 302.87 | 297.56 | SL hit (close<static) qty=1.00 sl=296.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 281.25 | 294.53 | 294.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 273.85 | 294.33 | 294.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 286.90 | 286.86 | 290.23 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-20 15:15:00 | 285.45 | 286.83 | 290.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:15:00 | 282.60 | 286.79 | 290.13 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 268.47 | 286.09 | 289.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-01-28 10:15:00 | 254.34 | 281.81 | 286.91 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 288.25 | 279.50 | 285.19 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 267.80 | 280.11 | 285.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 267.95 | 279.99 | 285.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-05 10:15:00 | 290.50 | 280.00 | 284.86 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-06 13:15:00 | 280.30 | 280.65 | 284.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:15:00 | 279.50 | 280.64 | 284.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-02-14 12:15:00 | 251.55 | 275.76 | 281.55 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-13 13:15:00 | 281.45 | 268.46 | 273.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 279.94 | 268.57 | 273.16 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 290.35 | 270.87 | 273.96 | SL hit (close>static) qty=1.00 sl=288.75 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 15:15:00 | 303.10 | 276.69 | 276.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 304.59 | 279.01 | 277.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.40 | 283.57 | 280.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 275.40 | 283.57 | 280.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 275.40 | 283.57 | 280.45 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 293.00 | 282.61 | 280.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 292.55 | 282.71 | 280.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-05-12 09:15:00 | 321.80 | 300.60 | 292.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.43 | 381.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.74 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.89 | 381.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.68 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 406.95 | 407.55 | 400.13 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-30 15:15:00 | 410.05 | 407.66 | 400.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:15:00 | 414.00 | 407.72 | 400.54 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-31 12:15:00 | 420.40 | 407.90 | 400.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 13:15:00 | 424.70 | 408.06 | 400.85 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-06 12:15:00 | 410.60 | 409.84 | 402.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 13:15:00 | 410.35 | 409.84 | 402.52 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 10:15:00 | 411.30 | 409.81 | 402.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 411.30 | 409.82 | 402.70 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | SL hit (close<ema400) qty=1.00 sl=408.63 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | SL hit (close<ema400) qty=1.00 sl=408.63 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | SL hit (close<ema400) qty=1.00 sl=408.63 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | SL hit (close<ema400) qty=1.00 sl=408.63 alert=retest1 |
| Cross detected — sustain check pending | 2025-11-25 11:15:00 | 411.00 | 415.34 | 408.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 411.25 | 415.30 | 408.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 411.85 | 415.13 | 408.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 411.90 | 415.10 | 408.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-28 14:15:00 | 411.60 | 414.83 | 409.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 412.00 | 414.81 | 409.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.58 | 409.42 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.58 | 409.42 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.58 | 409.42 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.95 | 405.67 | 405.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.85 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.69 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 400.55 | 400.43 | 402.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 400.25 | 400.43 | 402.67 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 404.35 | 400.46 | 402.66 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 400.85 | 400.54 | 402.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 398.70 | 400.53 | 402.65 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 404.85 | 399.51 | 401.80 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.00 | 403.78 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.30 | 408.90 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 443.83 | 432.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 432.80 | 443.42 | 433.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.42 | 433.03 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 440.05 | 442.07 | 432.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 439.50 | 442.05 | 432.92 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.41 | 433.07 | SL hit (close<static) qty=1.00 sl=425.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 438.55 | 429.77 | 428.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 440.35 | 429.88 | 428.83 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 439.95 | 439.32 | 434.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-28 10:15:00 | 437.20 | 439.29 | 434.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 438.70 | 439.12 | 434.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 438.85 | 439.12 | 434.94 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 437.65 | 439.08 | 435.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 437.50 | 439.07 | 435.02 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.97 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-04 09:15:00 | 436.95 | 438.40 | 434.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-04 10:15:00 | 435.10 | 438.37 | 434.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 436.40 | 437.77 | 434.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 438.00 | 437.78 | 434.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-05-07 10:15:00 | 436.30 | 437.72 | 434.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 437.85 | 437.72 | 434.86 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-25 10:15:00 | 289.80 | 2024-09-27 14:15:00 | 293.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-09-27 12:15:00 | 290.20 | 2024-09-27 14:15:00 | 293.75 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-09-30 10:15:00 | 286.20 | 2024-10-25 10:15:00 | 257.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-30 14:15:00 | 288.70 | 2024-11-06 09:15:00 | 292.10 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-11-04 10:15:00 | 281.45 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2024-11-13 10:15:00 | 284.20 | 2024-11-25 09:15:00 | 294.75 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-12-20 11:15:00 | 300.35 | 2024-12-20 13:15:00 | 293.35 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest1 | 2025-01-21 09:15:00 | 282.60 | 2025-01-22 10:15:00 | 268.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-21 09:15:00 | 282.60 | 2025-01-28 10:15:00 | 254.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-03 10:15:00 | 267.95 | 2025-02-05 10:15:00 | 290.50 | STOP_HIT | 1.00 | -8.42% |
| SELL | retest2 | 2025-02-06 14:15:00 | 279.50 | 2025-02-14 12:15:00 | 251.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-13 14:15:00 | 279.94 | 2025-03-19 11:15:00 | 290.35 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-04-15 10:15:00 | 292.55 | 2025-05-12 09:15:00 | 321.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-10-31 09:15:00 | 414.00 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-10-31 13:15:00 | 424.70 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2025-11-06 13:15:00 | 410.35 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-11-07 11:15:00 | 411.30 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-25 12:15:00 | 411.25 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-26 10:15:00 | 411.90 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-28 15:15:00 | 412.00 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-24 14:15:00 | 400.25 | 2025-12-26 09:15:00 | 404.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-26 13:15:00 | 398.70 | 2026-01-02 09:15:00 | 404.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-17 14:15:00 | 439.50 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.67% |
