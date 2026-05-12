# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 564.90
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 0.26% / 1.91%
- **Sum % (uncompounded):** 1.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.26% | 1.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.26% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 0.26% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 00:00:00 | 719.45 | 656.77 | 688.84 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=17.50 |
| Stop hit — per-position SL triggered | 2023-10-13 00:00:00 | 693.20 | 659.65 | 696.38 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 707.35 | 663.12 | 678.35 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=14.60 |
| Stop hit — per-position SL triggered | 2023-12-04 00:00:00 | 720.85 | 666.91 | 694.44 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 964.85 | 773.00 | 942.28 | Stage2 pullback-breakout RSI=56 vol=2.5x ATR=32.00 |
| Stop hit — per-position SL triggered | 2024-02-29 00:00:00 | 916.85 | 779.66 | 941.95 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 970.15 | 806.06 | 929.53 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=27.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 1024.92 | 810.08 | 944.04 | T1 booked 50% @ 1024.92 |
| Stop hit — per-position SL triggered | 2024-04-18 00:00:00 | 992.95 | 827.83 | 987.28 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-05 00:00:00 | 719.45 | 2023-10-13 00:00:00 | 693.20 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest1 | 2023-11-17 00:00:00 | 707.35 | 2023-12-04 00:00:00 | 720.85 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest1 | 2024-02-23 00:00:00 | 964.85 | 2024-02-29 00:00:00 | 916.85 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest1 | 2024-04-01 00:00:00 | 970.15 | 2024-04-03 00:00:00 | 1024.92 | PARTIAL | 0.50 | 5.65% |
| BUY | retest1 | 2024-04-01 00:00:00 | 970.15 | 2024-04-18 00:00:00 | 992.95 | STOP_HIT | 0.50 | 2.35% |
