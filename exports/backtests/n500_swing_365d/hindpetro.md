# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 387.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.86% / 5.10%
- **Sum % (uncompounded):** 19.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.86% | 19.3% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.86% | 19.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 3.86% | 19.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 406.25 | 386.63 | 398.81 | Stage2 pullback-breakout RSI=55 vol=2.7x ATR=11.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 05:30:00 | 429.99 | 387.69 | 405.22 | T1 booked 50% @ 429.99 |
| Target hit | 2025-07-18 05:30:00 | 430.60 | 394.96 | 431.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 05:30:00 | 468.85 | 408.20 | 446.07 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=11.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 05:30:00 | 492.76 | 414.41 | 466.92 | T1 booked 50% @ 492.76 |
| Stop hit — per-position SL triggered | 2025-11-14 05:30:00 | 481.25 | 415.78 | 469.93 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-02-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 05:30:00 | 453.25 | 433.11 | 443.53 | Stage2 pullback-breakout RSI=53 vol=2.2x ATR=14.97 |
| Stop hit — per-position SL triggered | 2026-02-16 05:30:00 | 452.05 | 435.40 | 451.86 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 406.25 | 2025-06-27 05:30:00 | 429.99 | PARTIAL | 0.50 | 5.84% |
| BUY | retest1 | 2025-06-24 05:30:00 | 406.25 | 2025-07-18 05:30:00 | 430.60 | TARGET_HIT | 0.50 | 5.99% |
| BUY | retest1 | 2025-10-29 05:30:00 | 468.85 | 2025-11-12 05:30:00 | 492.76 | PARTIAL | 0.50 | 5.10% |
| BUY | retest1 | 2025-10-29 05:30:00 | 468.85 | 2025-11-14 05:30:00 | 481.25 | STOP_HIT | 0.50 | 2.64% |
| BUY | retest1 | 2026-02-02 05:30:00 | 453.25 | 2026-02-16 05:30:00 | 452.05 | STOP_HIT | 1.00 | -0.26% |
