# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 674.50
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 3.16% / 5.43%
- **Sum % (uncompounded):** 12.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.16% | 12.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.16% | 12.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.16% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 405.65 | 303.29 | 386.14 | Stage2 pullback-breakout RSI=68 vol=2.2x ATR=11.82 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 387.91 | 306.84 | 388.46 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-08-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 00:00:00 | 408.70 | 314.25 | 391.89 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=11.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 00:00:00 | 430.88 | 317.57 | 400.99 | T1 booked 50% @ 430.88 |
| Target hit | 2024-10-07 00:00:00 | 450.85 | 354.58 | 460.16 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 520.10 | 375.36 | 479.73 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=16.99 |
| Stop hit — per-position SL triggered | 2024-11-14 00:00:00 | 526.80 | 389.48 | 507.22 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 00:00:00 | 405.65 | 2024-08-05 00:00:00 | 387.91 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest1 | 2024-08-19 00:00:00 | 408.70 | 2024-08-22 00:00:00 | 430.88 | PARTIAL | 0.50 | 5.43% |
| BUY | retest1 | 2024-08-19 00:00:00 | 408.70 | 2024-10-07 00:00:00 | 450.85 | TARGET_HIT | 0.50 | 10.31% |
| BUY | retest1 | 2024-10-31 00:00:00 | 520.10 | 2024-11-14 00:00:00 | 526.80 | STOP_HIT | 1.00 | 1.29% |
