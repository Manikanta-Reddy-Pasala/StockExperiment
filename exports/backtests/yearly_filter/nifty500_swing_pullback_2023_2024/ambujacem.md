# Ambuja Cements Ltd. (AMBUJACEM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 434.60
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 3.33% / 2.87%
- **Sum % (uncompounded):** 9.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.33% | 10.0% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.33% | 10.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.33% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 560.15 | 461.13 | 528.85 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=17.86 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 576.20 | 471.08 | 552.31 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 601.50 | 504.21 | 590.33 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=17.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 636.98 | 507.72 | 598.96 | T1 booked 50% @ 636.98 |
| Target hit | 2024-04-12 00:00:00 | 608.85 | 515.64 | 611.48 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-25 00:00:00 | 560.15 | 2024-02-09 00:00:00 | 576.20 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest1 | 2024-03-27 00:00:00 | 601.50 | 2024-04-02 00:00:00 | 636.98 | PARTIAL | 0.50 | 5.90% |
| BUY | retest1 | 2024-03-27 00:00:00 | 601.50 | 2024-04-12 00:00:00 | 608.85 | TARGET_HIT | 0.50 | 1.22% |
