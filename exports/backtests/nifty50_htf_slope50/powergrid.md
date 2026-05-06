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
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 18 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 9
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -1.56% / -1.51%
- **Sum % (uncompounded):** -15.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.34% | -7.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.34% | -7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.23% | -8.6% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.09% | -3.3% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.34% | -5.4% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.71% | -10.3% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.34% | -5.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 15:15:00)

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

### Cycle 2 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.16 | 288.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-04 13:15:00)

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

### Cycle 4 — BUY (started 2026-02-10 11:15:00)

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
