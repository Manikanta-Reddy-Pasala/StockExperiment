# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 9200.50
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
- **Avg / median % per leg:** 2.09% / 7.30%
- **Sum % (uncompounded):** 8.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.09% | 8.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.09% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.09% | 8.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 00:00:00 | 5589.50 | 4750.00 | 5315.75 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=205.37 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 5281.44 | 4767.95 | 5323.75 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 5369.05 | 4791.91 | 5301.54 | Stage2 pullback-breakout RSI=54 vol=2.2x ATR=195.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 00:00:00 | 5760.84 | 4848.36 | 5442.43 | T1 booked 50% @ 5760.84 |
| Target hit | 2024-10-07 00:00:00 | 6060.70 | 5214.92 | 6294.77 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 00:00:00 | 5663.70 | 5220.32 | 5099.89 | Stage2 pullback-breakout RSI=68 vol=5.2x ATR=237.63 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 5307.25 | 5221.48 | 5131.66 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 00:00:00 | 5589.50 | 2024-08-05 00:00:00 | 5281.44 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest1 | 2024-08-12 00:00:00 | 5369.05 | 2024-08-23 00:00:00 | 5760.84 | PARTIAL | 0.50 | 7.30% |
| BUY | retest1 | 2024-08-12 00:00:00 | 5369.05 | 2024-10-07 00:00:00 | 6060.70 | TARGET_HIT | 0.50 | 12.88% |
| BUY | retest1 | 2024-12-18 00:00:00 | 5663.70 | 2024-12-20 00:00:00 | 5307.25 | STOP_HIT | 1.00 | -6.29% |
