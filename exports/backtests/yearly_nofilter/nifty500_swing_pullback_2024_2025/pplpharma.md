# Piramal Pharma Ltd. (PPLPHARMA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 176.77
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
- **Avg / median % per leg:** 5.53% / 6.63%
- **Sum % (uncompounded):** 33.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 5.53% | 33.2% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 5.53% | 33.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 5.53% | 33.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 00:00:00 | 166.55 | 136.79 | 154.69 | Stage2 pullback-breakout RSI=67 vol=3.8x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 00:00:00 | 177.59 | 140.11 | 165.72 | T1 booked 50% @ 177.59 |
| Target hit | 2024-10-07 00:00:00 | 218.02 | 163.63 | 220.80 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 00:00:00 | 231.52 | 167.10 | 222.01 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=9.13 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 217.82 | 169.92 | 222.96 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-10-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 00:00:00 | 255.84 | 171.25 | 225.66 | Stage2 pullback-breakout RSI=67 vol=9.4x ATR=11.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-01 00:00:00 | 279.83 | 176.28 | 241.00 | T1 booked 50% @ 279.83 |
| Stop hit — per-position SL triggered | 2024-11-12 00:00:00 | 255.84 | 183.06 | 258.21 | SL hit (bars_held=13) |

### Cycle 4 — BUY (started 2024-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 00:00:00 | 268.75 | 190.31 | 254.55 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=13.99 |
| Stop hit — per-position SL triggered | 2024-12-13 00:00:00 | 247.76 | 197.13 | 258.02 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-24 00:00:00 | 166.55 | 2024-08-07 00:00:00 | 177.59 | PARTIAL | 0.50 | 6.63% |
| BUY | retest1 | 2024-07-24 00:00:00 | 166.55 | 2024-10-07 00:00:00 | 218.02 | TARGET_HIT | 0.50 | 30.90% |
| BUY | retest1 | 2024-10-15 00:00:00 | 231.52 | 2024-10-22 00:00:00 | 217.82 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest1 | 2024-10-24 00:00:00 | 255.84 | 2024-11-01 00:00:00 | 279.83 | PARTIAL | 0.50 | 9.38% |
| BUY | retest1 | 2024-10-24 00:00:00 | 255.84 | 2024-11-12 00:00:00 | 255.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 00:00:00 | 268.75 | 2024-12-13 00:00:00 | 247.76 | STOP_HIT | 1.00 | -7.81% |
