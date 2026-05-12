# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 73.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.78% / 0.00%
- **Sum % (uncompounded):** 8.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.78% | 8.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.78% | 8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.78% | 8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 00:00:00 | 85.59 | 80.11 | 84.91 | Stage2 pullback-breakout RSI=52 vol=2.1x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 00:00:00 | 90.21 | 80.39 | 85.78 | T1 booked 50% @ 90.21 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 85.59 | 80.81 | 86.99 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-07-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 00:00:00 | 97.50 | 80.97 | 87.99 | Stage2 pullback-breakout RSI=66 vol=6.1x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 00:00:00 | 105.96 | 81.41 | 90.70 | T1 booked 50% @ 105.96 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 97.50 | 82.44 | 95.27 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 94.92 | 85.30 | 92.43 | Stage2 pullback-breakout RSI=54 vol=6.6x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-09-18 00:00:00 | 90.02 | 85.49 | 92.18 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-09 00:00:00 | 85.59 | 2024-07-15 00:00:00 | 90.21 | PARTIAL | 0.50 | 5.40% |
| BUY | retest1 | 2024-07-09 00:00:00 | 85.59 | 2024-07-23 00:00:00 | 85.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 00:00:00 | 97.50 | 2024-07-26 00:00:00 | 105.96 | PARTIAL | 0.50 | 8.67% |
| BUY | retest1 | 2024-07-24 00:00:00 | 97.50 | 2024-08-02 00:00:00 | 97.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 00:00:00 | 94.92 | 2024-09-18 00:00:00 | 90.02 | STOP_HIT | 1.00 | -5.16% |
