# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 879.55
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.15% / 1.44%
- **Sum % (uncompounded):** -4.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.15% | -4.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.15% | -4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -1.15% | -4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 05:30:00 | 780.10 | 761.29 | 763.52 | Stage2 pullback-breakout RSI=56 vol=3.8x ATR=17.39 |
| Stop hit — per-position SL triggered | 2025-08-26 05:30:00 | 754.01 | 761.42 | 763.37 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 05:30:00 | 789.65 | 760.71 | 765.37 | Stage2 pullback-breakout RSI=61 vol=5.6x ATR=18.86 |
| Stop hit — per-position SL triggered | 2025-10-17 05:30:00 | 801.05 | 764.25 | 786.10 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 05:30:00 | 939.15 | 827.97 | 917.44 | Stage2 pullback-breakout RSI=58 vol=2.5x ATR=29.11 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 895.48 | 829.52 | 915.42 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-04-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 05:30:00 | 886.10 | 844.11 | 848.01 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=31.93 |
| Stop hit — per-position SL triggered | 2026-05-06 05:30:00 | 903.30 | 848.76 | 875.24 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-20 05:30:00 | 780.10 | 2025-08-26 05:30:00 | 754.01 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest1 | 2025-10-03 05:30:00 | 789.65 | 2025-10-17 05:30:00 | 801.05 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest1 | 2026-01-19 05:30:00 | 939.15 | 2026-01-21 05:30:00 | 895.48 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest1 | 2026-04-21 05:30:00 | 886.10 | 2026-05-06 05:30:00 | 903.30 | STOP_HIT | 1.00 | 1.94% |
