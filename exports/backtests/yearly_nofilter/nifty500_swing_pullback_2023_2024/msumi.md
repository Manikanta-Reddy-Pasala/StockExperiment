# Motherson Sumi Wiring India Ltd. (MSUMI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 41.62
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.77% / 4.82%
- **Sum % (uncompounded):** 10.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.77% | 10.6% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.77% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 1.77% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 41.40 | 37.89 | 39.91 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 43.40 | 37.95 | 40.30 | T1 booked 50% @ 43.40 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 41.40 | 38.39 | 42.12 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 41.07 | 39.39 | 40.19 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 40.20 | 39.52 | 40.52 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 42.33 | 40.04 | 41.50 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 00:00:00 | 44.77 | 40.14 | 42.15 | T1 booked 50% @ 44.77 |
| Target hit | 2024-02-12 00:00:00 | 44.37 | 40.59 | 44.40 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 47.20 | 42.26 | 45.45 | Stage2 pullback-breakout RSI=61 vol=2.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-05-02 00:00:00 | 45.93 | 42.68 | 46.13 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 41.40 | 2023-09-01 00:00:00 | 43.40 | PARTIAL | 0.50 | 4.83% |
| BUY | retest1 | 2023-08-31 00:00:00 | 41.40 | 2023-09-13 00:00:00 | 41.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 00:00:00 | 41.07 | 2023-12-14 00:00:00 | 40.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest1 | 2024-01-30 00:00:00 | 42.33 | 2024-02-01 00:00:00 | 44.77 | PARTIAL | 0.50 | 5.77% |
| BUY | retest1 | 2024-01-30 00:00:00 | 42.33 | 2024-02-12 00:00:00 | 44.37 | TARGET_HIT | 0.50 | 4.82% |
| BUY | retest1 | 2024-04-16 00:00:00 | 47.20 | 2024-05-02 00:00:00 | 45.93 | STOP_HIT | 1.00 | -2.69% |
