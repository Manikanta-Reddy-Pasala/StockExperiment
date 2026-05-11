# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 144.69
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
- **Avg / median % per leg:** 21.13% / 3.99%
- **Sum % (uncompounded):** 84.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 21.13% | 84.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 21.13% | 84.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 21.13% | 84.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 05:30:00 | 94.00 | 85.60 | 91.96 | Stage2 pullback-breakout RSI=58 vol=2.9x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-09-12 05:30:00 | 91.63 | 85.75 | 92.20 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-11-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 05:30:00 | 92.45 | 87.02 | 89.67 | Stage2 pullback-breakout RSI=59 vol=4.1x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 05:30:00 | 96.13 | 87.10 | 90.18 | T1 booked 50% @ 96.13 |
| Target hit | 2024-02-27 05:30:00 | 173.25 | 117.31 | 174.22 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 05:30:00 | 176.75 | 134.36 | 169.98 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=5.14 |
| Stop hit — per-position SL triggered | 2024-04-30 05:30:00 | 169.04 | 134.70 | 169.87 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-08 05:30:00 | 94.00 | 2023-09-12 05:30:00 | 91.63 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest1 | 2023-11-01 05:30:00 | 92.45 | 2023-11-02 05:30:00 | 96.13 | PARTIAL | 0.50 | 3.99% |
| BUY | retest1 | 2023-11-01 05:30:00 | 92.45 | 2024-02-27 05:30:00 | 173.25 | TARGET_HIT | 0.50 | 87.40% |
| BUY | retest1 | 2024-04-29 05:30:00 | 176.75 | 2024-04-30 05:30:00 | 169.04 | STOP_HIT | 1.00 | -4.36% |
