# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 518.70
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 3
- **Avg / median % per leg:** 2.58% / 4.63%
- **Sum % (uncompounded):** 15.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 2 | 3 | 2.58% | 15.5% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 2 | 3 | 2.58% | 15.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 2 | 3 | 2.58% | 15.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 00:00:00 | 389.95 | 383.03 | 385.07 | Stage2 pullback-breakout RSI=53 vol=2.5x ATR=9.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 00:00:00 | 408.01 | 384.04 | 391.69 | T1 booked 50% @ 408.01 |
| Target hit | 2023-08-08 00:00:00 | 394.05 | 385.24 | 397.33 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 407.10 | 386.42 | 393.85 | Stage2 pullback-breakout RSI=60 vol=3.5x ATR=10.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 00:00:00 | 427.76 | 388.53 | 405.55 | T1 booked 50% @ 427.76 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 407.10 | 390.60 | 415.07 | SL hit (bars_held=13) |

### Cycle 3 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 445.15 | 402.11 | 429.69 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=10.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 00:00:00 | 466.12 | 403.83 | 437.76 | T1 booked 50% @ 466.12 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 445.15 | 408.46 | 451.42 | SL hit (bars_held=11) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-20 00:00:00 | 389.95 | 2023-07-31 00:00:00 | 408.01 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2023-07-20 00:00:00 | 389.95 | 2023-08-08 00:00:00 | 394.05 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2023-09-04 00:00:00 | 407.10 | 2023-09-14 00:00:00 | 427.76 | PARTIAL | 0.50 | 5.07% |
| BUY | retest1 | 2023-09-04 00:00:00 | 407.10 | 2023-09-22 00:00:00 | 407.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-20 00:00:00 | 445.15 | 2023-11-23 00:00:00 | 466.12 | PARTIAL | 0.50 | 4.71% |
| BUY | retest1 | 2023-11-20 00:00:00 | 445.15 | 2023-12-06 00:00:00 | 445.15 | STOP_HIT | 0.50 | 0.00% |
