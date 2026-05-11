# Zee Entertainment Enterprises Ltd. (ZEEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 91.57
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
- **Avg / median % per leg:** -0.81% / 0.00%
- **Sum % (uncompounded):** -3.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 00:00:00 | 267.70 | 237.12 | 256.54 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=9.54 |
| Stop hit — per-position SL triggered | 2023-11-10 00:00:00 | 253.38 | 238.92 | 259.53 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 266.55 | 240.58 | 254.43 | Stage2 pullback-breakout RSI=59 vol=3.0x ATR=8.98 |
| Stop hit — per-position SL triggered | 2023-12-05 00:00:00 | 253.09 | 240.84 | 254.29 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 271.60 | 241.31 | 256.27 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=9.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 00:00:00 | 291.03 | 242.59 | 263.71 | T1 booked 50% @ 291.03 |
| Stop hit — per-position SL triggered | 2023-12-14 00:00:00 | 271.60 | 243.36 | 266.86 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-01 00:00:00 | 267.70 | 2023-11-10 00:00:00 | 253.38 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2023-12-01 00:00:00 | 266.55 | 2023-12-05 00:00:00 | 253.09 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest1 | 2023-12-07 00:00:00 | 271.60 | 2023-12-12 00:00:00 | 291.03 | PARTIAL | 0.50 | 7.15% |
| BUY | retest1 | 2023-12-07 00:00:00 | 271.60 | 2023-12-14 00:00:00 | 271.60 | STOP_HIT | 0.50 | 0.00% |
