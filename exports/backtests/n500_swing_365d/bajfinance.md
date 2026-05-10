# Bajaj Finance Ltd. (BAJFINANCE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 955.35
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 2.74% / 4.62%
- **Sum % (uncompounded):** 16.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.74% | 16.4% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.74% | 16.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.74% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 05:30:00 | 905.20 | 854.41 | 890.22 | Stage2 pullback-breakout RSI=53 vol=2.0x ATR=22.07 |
| Stop hit — per-position SL triggered | 2025-08-28 05:30:00 | 872.10 | 856.84 | 889.70 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2025-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 05:30:00 | 934.75 | 858.85 | 893.81 | Stage2 pullback-breakout RSI=64 vol=2.9x ATR=21.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 05:30:00 | 977.93 | 864.87 | 926.20 | T1 booked 50% @ 977.93 |
| Target hit | 2025-10-31 05:30:00 | 1042.80 | 912.53 | 1049.34 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 05:30:00 | 1033.80 | 930.83 | 1022.75 | Stage2 pullback-breakout RSI=52 vol=1.6x ATR=23.76 |
| Stop hit — per-position SL triggered | 2025-12-11 05:30:00 | 1006.40 | 939.70 | 1022.44 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-02-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 05:30:00 | 981.70 | 948.02 | 951.39 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=25.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 05:30:00 | 1032.01 | 953.39 | 987.93 | T1 booked 50% @ 1032.01 |
| Target hit | 2026-02-27 05:30:00 | 995.90 | 956.48 | 998.69 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-18 05:30:00 | 905.20 | 2025-08-28 05:30:00 | 872.10 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2025-09-04 05:30:00 | 934.75 | 2025-09-12 05:30:00 | 977.93 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2025-09-04 05:30:00 | 934.75 | 2025-10-31 05:30:00 | 1042.80 | TARGET_HIT | 0.50 | 11.56% |
| BUY | retest1 | 2025-11-27 05:30:00 | 1033.80 | 2025-12-11 05:30:00 | 1006.40 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2026-02-06 05:30:00 | 981.70 | 2026-02-20 05:30:00 | 1032.01 | PARTIAL | 0.50 | 5.12% |
| BUY | retest1 | 2026-02-06 05:30:00 | 981.70 | 2026-02-27 05:30:00 | 995.90 | TARGET_HIT | 0.50 | 1.45% |
