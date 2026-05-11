# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1760.40
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
- **Avg / median % per leg:** 7.71% / 4.59%
- **Sum % (uncompounded):** 30.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 7.71% | 30.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 7.71% | 30.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 7.71% | 30.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 05:30:00 | 818.70 | 772.75 | 796.18 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=17.97 |
| Stop hit — per-position SL triggered | 2023-11-22 05:30:00 | 791.75 | 775.98 | 802.12 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-11-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 05:30:00 | 837.70 | 776.95 | 804.24 | Stage2 pullback-breakout RSI=65 vol=4.4x ATR=19.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 05:30:00 | 876.19 | 779.51 | 817.22 | T1 booked 50% @ 876.19 |
| Target hit | 2024-01-24 05:30:00 | 1120.60 | 877.41 | 1134.82 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 05:30:00 | 1404.15 | 1029.19 | 1309.50 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=39.65 |
| Stop hit — per-position SL triggered | 2024-04-08 05:30:00 | 1344.67 | 1042.63 | 1329.58 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-08 05:30:00 | 818.70 | 2023-11-22 05:30:00 | 791.75 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2023-11-28 05:30:00 | 837.70 | 2023-12-04 05:30:00 | 876.19 | PARTIAL | 0.50 | 4.59% |
| BUY | retest1 | 2023-11-28 05:30:00 | 837.70 | 2024-01-24 05:30:00 | 1120.60 | TARGET_HIT | 0.50 | 33.77% |
| BUY | retest1 | 2024-04-02 05:30:00 | 1404.15 | 2024-04-08 05:30:00 | 1344.67 | STOP_HIT | 1.00 | -4.24% |
