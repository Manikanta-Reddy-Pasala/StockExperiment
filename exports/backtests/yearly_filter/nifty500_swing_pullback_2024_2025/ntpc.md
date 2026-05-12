# NTPC Ltd. (NTPC)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 402.15
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 2.69% / 2.10%
- **Sum % (uncompounded):** 8.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.69% | 8.1% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.69% | 8.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.69% | 8.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 377.15 | 319.07 | 364.12 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=11.53 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 377.10 | 324.47 | 371.84 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 00:00:00 | 392.60 | 328.68 | 376.20 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=11.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 00:00:00 | 416.10 | 332.22 | 386.46 | T1 booked 50% @ 416.10 |
| Stop hit — per-position SL triggered | 2024-08-12 00:00:00 | 400.85 | 338.45 | 400.53 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 377.15 | 2024-07-11 00:00:00 | 377.10 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest1 | 2024-07-24 00:00:00 | 392.60 | 2024-07-31 00:00:00 | 416.10 | PARTIAL | 0.50 | 5.99% |
| BUY | retest1 | 2024-07-24 00:00:00 | 392.60 | 2024-08-12 00:00:00 | 400.85 | STOP_HIT | 0.50 | 2.10% |
