# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 960.70
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.78% / -3.71%
- **Sum % (uncompounded):** -11.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.78% | -11.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.78% | -11.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.78% | -11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 339.30 | 299.04 | 330.76 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=8.39 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 326.72 | 300.10 | 331.72 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-10-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 00:00:00 | 338.50 | 306.35 | 334.35 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=7.75 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 326.87 | 307.93 | 333.70 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-05-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 00:00:00 | 461.40 | 385.80 | 439.40 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=12.93 |
| Stop hit — per-position SL triggered | 2024-05-10 00:00:00 | 442.01 | 388.41 | 443.52 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-08 00:00:00 | 339.30 | 2023-09-13 00:00:00 | 326.72 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest1 | 2023-10-13 00:00:00 | 338.50 | 2023-10-23 00:00:00 | 326.87 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2024-05-06 00:00:00 | 461.40 | 2024-05-10 00:00:00 | 442.01 | STOP_HIT | 1.00 | -4.20% |
