# NSE:POWERGRID-EQ (NSE:POWERGRID-EQ)

## Backtest Summary

- **Window:** 2024-04-04 09:15:00 → 2026-05-08 15:15:00 (3612 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 25 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 11 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 2 / 16 / 4
- **Avg / median % per leg:** 0.82% / -1.40%
- **Sum % (uncompounded):** 18.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 2 | 9 | 0 | -0.38% | -4.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.41% | 9.9% |
| SELL (all) | 11 | 8 | 72.7% | 0 | 7 | 4 | 2.02% | 22.2% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.22% | 19.3% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.58% | 2.9% |
| retest1 (combined) | 10 | 6 | 60.0% | 0 | 7 | 3 | 0.53% | 5.3% |
| retest2 (combined) | 12 | 4 | 33.3% | 2 | 9 | 1 | 1.06% | 12.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.10 | 294.84 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 299.50 | 294.46 | 294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-21 11:15:00 | 296.30 | 295.76 | 295.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:15:00 | 297.00 | 295.77 | 295.19 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 11:15:00 | 297.15 | 295.82 | 295.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 297.75 | 295.84 | 295.24 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.15 | 295.48 | SL hit (close<static) qty=1.00 sl=293.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.15 | 295.48 | SL hit (close<static) qty=1.00 sl=293.95 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 290.30 | 290.96 | 292.57 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 15:15:00 | 290.70 | 290.95 | 292.56 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 289.50 | 290.94 | 292.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:15:00 | 289.15 | 290.92 | 292.53 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 274.69 | 288.03 | 290.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 287.15 | 286.53 | 289.64 | SL hit (close>ema200) qty=0.50 sl=286.53 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 284.00 | 286.95 | 288.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 284.20 | 286.92 | 288.32 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 289.45 | 285.66 | 287.41 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 285.15 | 285.68 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 285.55 | 285.67 | 287.40 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 289.55 | 285.75 | 287.38 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 285.55 | 285.97 | 287.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 11:15:00 | 285.80 | 285.97 | 287.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 13:15:00 | 284.70 | 285.96 | 287.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 285.25 | 285.95 | 287.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 285.40 | 285.91 | 287.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 14:15:00 | 286.05 | 285.91 | 287.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.33 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 285.60 | 286.15 | 287.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-13 14:15:00 | 285.75 | 286.15 | 287.35 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.16 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 15:15:00 | 267.00 | 268.48 | 273.87 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 267.50 | 268.45 | 273.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 13:15:00 | 266.80 | 268.44 | 273.54 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 267.40 | 267.61 | 271.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 268.05 | 267.62 | 271.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 254.65 | 266.35 | 270.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.65 | 263.47 | 268.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.46 | 263.47 | 268.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.22 | 282.89 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.85 | 290.24 | 282.94 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.00 | 290.26 | 282.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.32 | 283.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.10 | 292.98 | 285.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.02 | 285.61 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.23 | 288.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.24 | 288.70 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.23 | 289.23 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 291.80 | 294.68 | 289.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.51 | 289.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.50 | 289.20 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-20 10:15:00 | 320.98 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-20 10:15:00 | 322.80 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:15:00 | 308.00 | 2025-05-14 12:15:00 | 294.95 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-05-15 14:15:00 | 299.45 | 2025-05-22 09:15:00 | 289.35 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-23 12:15:00 | 298.05 | 2025-05-27 09:15:00 | 293.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-09 13:15:00 | 300.15 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-22 12:15:00 | 297.75 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2025-08-19 10:15:00 | 289.15 | 2025-08-28 14:15:00 | 274.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-19 10:15:00 | 289.15 | 2025-09-02 10:15:00 | 287.15 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2025-09-25 15:15:00 | 284.20 | 2025-10-03 14:15:00 | 289.45 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-10-06 10:15:00 | 285.55 | 2025-10-07 09:15:00 | 289.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-08 14:15:00 | 285.25 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-12 09:15:00 | 254.65 | PARTIAL | 0.50 | 4.63% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.80 | 2026-01-20 14:15:00 | 253.65 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-01-20 14:15:00 | 253.46 | PARTIAL | 0.50 | 5.44% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.80 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-06 09:15:00 | 291.80 | 2026-04-20 10:15:00 | 320.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:15:00 | 293.45 | 2026-04-20 10:15:00 | 322.80 | TARGET_HIT | 1.00 | 10.00% |
