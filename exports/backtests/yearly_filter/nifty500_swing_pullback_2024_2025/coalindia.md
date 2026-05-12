# Coal India Ltd. (COALINDIA)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 455.85
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.26% / 0.00%
- **Sum % (uncompounded):** -1.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.26% | -1.1% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.26% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.26% | -1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 00:00:00 | 500.05 | 413.71 | 484.60 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=12.33 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 481.55 | 419.33 | 490.26 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 509.85 | 421.69 | 492.72 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=15.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 00:00:00 | 540.25 | 425.69 | 503.25 | T1 booked 50% @ 540.25 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 509.85 | 427.47 | 505.42 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 00:00:00 | 516.10 | 453.25 | 501.77 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=11.41 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 498.99 | 455.26 | 502.50 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-11 00:00:00 | 500.05 | 2024-07-23 00:00:00 | 481.55 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest1 | 2024-07-26 00:00:00 | 509.85 | 2024-08-01 00:00:00 | 540.25 | PARTIAL | 0.50 | 5.96% |
| BUY | retest1 | 2024-07-26 00:00:00 | 509.85 | 2024-08-05 00:00:00 | 509.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 00:00:00 | 516.10 | 2024-10-04 00:00:00 | 498.99 | STOP_HIT | 1.00 | -3.32% |
