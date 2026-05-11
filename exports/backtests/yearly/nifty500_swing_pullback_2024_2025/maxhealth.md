# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1012.50
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
- **Avg / median % per leg:** 0.97% / 3.10%
- **Sum % (uncompounded):** 3.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.97% | 3.9% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.97% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 0.97% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 05:30:00 | 913.80 | 809.08 | 881.99 | Stage2 pullback-breakout RSI=61 vol=2.4x ATR=27.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 05:30:00 | 968.96 | 812.80 | 896.12 | T1 booked 50% @ 968.96 |
| Target hit | 2024-10-03 05:30:00 | 942.15 | 832.06 | 960.31 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-03-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 05:30:00 | 1135.85 | 983.03 | 1026.27 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=41.13 |
| Stop hit — per-position SL triggered | 2025-04-04 05:30:00 | 1137.65 | 996.74 | 1087.57 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-05-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 05:30:00 | 1156.00 | 1013.74 | 1103.17 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=41.68 |
| Stop hit — per-position SL triggered | 2025-05-09 05:30:00 | 1093.48 | 1017.19 | 1110.06 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-12 05:30:00 | 913.80 | 2024-09-17 05:30:00 | 968.96 | PARTIAL | 0.50 | 6.04% |
| BUY | retest1 | 2024-09-12 05:30:00 | 913.80 | 2024-10-03 05:30:00 | 942.15 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2025-03-20 05:30:00 | 1135.85 | 2025-04-04 05:30:00 | 1137.65 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest1 | 2025-05-06 05:30:00 | 1156.00 | 2025-05-09 05:30:00 | 1093.48 | STOP_HIT | 1.00 | -5.41% |
