# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 401.95
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
- **Avg / median % per leg:** 4.23% / 6.61%
- **Sum % (uncompounded):** 16.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.23% | 16.9% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.23% | 16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.23% | 16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 00:00:00 | 181.98 | 162.12 | 176.91 | Stage2 pullback-breakout RSI=53 vol=1.9x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 171.70 | 163.70 | 177.35 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 195.20 | 165.97 | 182.82 | Stage2 pullback-breakout RSI=63 vol=2.8x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 00:00:00 | 208.10 | 166.75 | 186.87 | T1 booked 50% @ 208.10 |
| Target hit | 2024-11-12 00:00:00 | 225.91 | 182.44 | 227.97 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 00:00:00 | 248.21 | 184.94 | 230.79 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=11.11 |
| Stop hit — per-position SL triggered | 2024-12-05 00:00:00 | 248.79 | 190.93 | 241.26 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-26 00:00:00 | 181.98 | 2024-09-09 00:00:00 | 171.70 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2024-09-25 00:00:00 | 195.20 | 2024-09-27 00:00:00 | 208.10 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2024-09-25 00:00:00 | 195.20 | 2024-11-12 00:00:00 | 225.91 | TARGET_HIT | 0.50 | 15.73% |
| BUY | retest1 | 2024-11-21 00:00:00 | 248.21 | 2024-12-05 00:00:00 | 248.79 | STOP_HIT | 1.00 | 0.23% |
