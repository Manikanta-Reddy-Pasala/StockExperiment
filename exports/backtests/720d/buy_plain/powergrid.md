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
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 19 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 1 / 10
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -2.25% / -1.86%
- **Sum % (uncompounded):** -24.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 0 | 11 | 0 | -2.25% | -24.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.53% | -10.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.53% | -10.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 306.00 | 285.86 | 285.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.65 | 286.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.85 | 300.83 | 295.30 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 308.15 | 300.85 | 295.50 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:15:00 | 308.00 | 300.92 | 295.57 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.16 | 296.08 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-14 12:15:00 | 294.95 | 301.10 | 296.08 | SL hit (close<ema400) qty=1.00 sl=296.08 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 299.30 | 300.60 | 296.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 299.45 | 300.59 | 296.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-21 13:15:00 | 298.15 | 300.60 | 296.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-21 14:15:00 | 296.05 | 300.55 | 296.62 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 289.35 | 300.40 | 296.58 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 298.00 | 299.70 | 296.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 298.05 | 299.68 | 296.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 293.25 | 299.52 | 296.49 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 12:15:00 | 299.30 | 296.25 | 295.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:15:00 | 300.15 | 296.29 | 295.44 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 295.10 | 296.71 | 295.72 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 10:15:00 | 298.15 | 292.79 | 293.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 298.70 | 292.85 | 293.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 295.40 | 293.91 | 294.16 | SL hit (close<static) qty=1.00 sl=295.60 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 294.80 | 293.92 | 294.16 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 296.00 | 294.00 | 294.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-07 11:15:00 | 295.90 | 294.02 | 294.19 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-07 14:15:00 | 296.00 | 294.05 | 294.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-07 15:15:00 | 295.45 | 294.06 | 294.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-08 09:15:00 | 296.75 | 294.09 | 294.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 297.55 | 294.12 | 294.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 299.50 | 294.46 | 294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 11:15:00 | 296.30 | 295.76 | 295.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:15:00 | 297.00 | 295.77 | 295.19 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 11:15:00 | 297.15 | 295.82 | 295.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 297.75 | 295.84 | 295.24 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.15 | 295.48 | SL hit (close<static) qty=1.00 sl=293.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 292.20 | 296.15 | 295.48 | SL hit (close<static) qty=1.00 sl=293.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-29 12:15:00 | 296.20 | 288.10 | 288.13 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 3 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.16 | EMA200 above EMA400 |

### Cycle 4 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.22 | 282.89 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.85 | 290.24 | 282.94 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.00 | 290.26 | 282.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.32 | 283.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.10 | 292.98 | 285.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.02 | 285.61 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.23 | 288.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.24 | 288.70 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.23 | 289.23 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 291.80 | 294.68 | 289.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.51 | 289.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.50 | 289.20 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:15:00 | 308.00 | 2025-05-14 12:15:00 | 294.95 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-05-15 14:15:00 | 299.45 | 2025-05-22 09:15:00 | 289.35 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-23 12:15:00 | 298.05 | 2025-05-27 09:15:00 | 293.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-09 13:15:00 | 300.15 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-06-27 11:15:00 | 298.70 | 2025-07-02 13:15:00 | 295.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-08 10:15:00 | 297.55 | 2025-07-09 10:15:00 | 299.15 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-22 12:15:00 | 297.75 | 2025-07-25 13:15:00 | 292.20 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.26% |
