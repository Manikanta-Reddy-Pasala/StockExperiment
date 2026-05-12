# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1787.80
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 1.17% / 0.00%
- **Sum % (uncompounded):** 3.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.17% | 3.5% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.17% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.17% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-03-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 00:00:00 | 1220.00 | 994.47 | 1164.59 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=29.76 |
| Stop hit — per-position SL triggered | 2024-04-02 00:00:00 | 1208.25 | 1016.24 | 1200.94 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 1265.75 | 1035.61 | 1214.98 | Stage2 pullback-breakout RSI=66 vol=2.7x ATR=28.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 00:00:00 | 1322.22 | 1043.72 | 1240.00 | T1 booked 50% @ 1322.22 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 1265.75 | 1062.35 | 1278.24 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-15 00:00:00 | 1220.00 | 2024-04-02 00:00:00 | 1208.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1265.75 | 2024-04-23 00:00:00 | 1322.22 | PARTIAL | 0.50 | 4.46% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1265.75 | 2024-05-03 00:00:00 | 1265.75 | STOP_HIT | 0.50 | 0.00% |
