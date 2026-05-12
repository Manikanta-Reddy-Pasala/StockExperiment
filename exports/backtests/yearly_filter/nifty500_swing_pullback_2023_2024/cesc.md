# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 184.41
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 9.80% / 5.57%
- **Sum % (uncompounded):** 39.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.80% | 39.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.80% | 39.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.80% | 39.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 92.55 | 78.76 | 88.89 | Stage2 pullback-breakout RSI=63 vol=3.2x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 88.74 | 79.09 | 89.13 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 90.50 | 79.85 | 87.16 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 95.54 | 80.50 | 89.02 | T1 booked 50% @ 95.54 |
| Target hit | 2024-01-23 00:00:00 | 130.60 | 96.63 | 134.53 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 143.15 | 98.20 | 135.43 | Stage2 pullback-breakout RSI=63 vol=2.6x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 133.77 | 101.36 | 137.38 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-17 00:00:00 | 92.55 | 2023-10-20 00:00:00 | 88.74 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest1 | 2023-11-08 00:00:00 | 90.50 | 2023-11-16 00:00:00 | 95.54 | PARTIAL | 0.50 | 5.57% |
| BUY | retest1 | 2023-11-08 00:00:00 | 90.50 | 2024-01-23 00:00:00 | 130.60 | TARGET_HIT | 0.50 | 44.31% |
| BUY | retest1 | 2024-01-30 00:00:00 | 143.15 | 2024-02-09 00:00:00 | 133.77 | STOP_HIT | 1.00 | -6.55% |
