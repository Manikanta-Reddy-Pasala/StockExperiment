# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 724.00
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
- **Avg / median % per leg:** 5.50% / 7.55%
- **Sum % (uncompounded):** 16.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 1 | 1 | 5.50% | 16.5% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 5.50% | 16.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 5.50% | 16.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-02-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 00:00:00 | 483.20 | 456.78 | 454.93 | Stage2 pullback-breakout RSI=60 vol=11.9x ATR=19.32 |
| Stop hit — per-position SL triggered | 2024-02-26 00:00:00 | 486.90 | 458.83 | 469.70 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 483.55 | 459.13 | 459.66 | Stage2 pullback-breakout RSI=61 vol=5.5x ATR=18.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 00:00:00 | 520.05 | 460.90 | 475.00 | T1 booked 50% @ 520.05 |
| Target hit | 2024-05-07 00:00:00 | 523.20 | 473.01 | 528.46 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-02-12 00:00:00 | 483.20 | 2024-02-26 00:00:00 | 486.90 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest1 | 2024-04-04 00:00:00 | 483.55 | 2024-04-10 00:00:00 | 520.05 | PARTIAL | 0.50 | 7.55% |
| BUY | retest1 | 2024-04-04 00:00:00 | 483.55 | 2024-05-07 00:00:00 | 523.20 | TARGET_HIT | 0.50 | 8.20% |
