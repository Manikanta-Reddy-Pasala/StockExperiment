# Coal India Ltd. (COALINDIA)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 456.40
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
- **Avg / median % per leg:** 1.34% / 0.00%
- **Sum % (uncompounded):** 6.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.34% | 6.7% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.34% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.34% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 444.05 | 398.75 | 419.99 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=10.67 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 428.05 | 399.92 | 424.53 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2026-03-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 05:30:00 | 449.40 | 405.68 | 429.53 | Stage2 pullback-breakout RSI=63 vol=3.6x ATR=11.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 05:30:00 | 472.64 | 407.73 | 437.24 | T1 booked 50% @ 472.64 |
| Stop hit — per-position SL triggered | 2026-03-18 05:30:00 | 449.40 | 409.83 | 445.00 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2026-04-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 05:30:00 | 467.00 | 418.39 | 448.51 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=11.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 05:30:00 | 490.96 | 419.62 | 454.35 | T1 booked 50% @ 490.96 |
| Stop hit — per-position SL triggered | 2026-05-07 05:30:00 | 467.00 | 421.69 | 460.11 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-28 05:30:00 | 444.05 | 2026-02-01 05:30:00 | 428.05 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2026-03-05 05:30:00 | 449.40 | 2026-03-12 05:30:00 | 472.64 | PARTIAL | 0.50 | 5.17% |
| BUY | retest1 | 2026-03-05 05:30:00 | 449.40 | 2026-03-18 05:30:00 | 449.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 05:30:00 | 467.00 | 2026-04-30 05:30:00 | 490.96 | PARTIAL | 0.50 | 5.13% |
| BUY | retest1 | 2026-04-28 05:30:00 | 467.00 | 2026-05-07 05:30:00 | 467.00 | STOP_HIT | 0.50 | 0.00% |
