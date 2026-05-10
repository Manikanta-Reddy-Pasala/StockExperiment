# AU Small Finance Bank Ltd. (AUBANK)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1050.40
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 7.58% / 4.37%
- **Sum % (uncompounded):** 37.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 7.58% | 37.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 7.58% | 37.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 7.58% | 37.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 05:30:00 | 731.65 | 693.33 | 718.43 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=15.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 05:30:00 | 763.61 | 696.93 | 730.84 | T1 booked 50% @ 763.61 |
| Target hit | 2026-01-13 05:30:00 | 971.95 | 813.50 | 987.99 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2026-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 05:30:00 | 1025.40 | 817.21 | 990.55 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=22.78 |
| Stop hit — per-position SL triggered | 2026-01-21 05:30:00 | 991.24 | 822.78 | 994.26 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-02-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 05:30:00 | 1022.05 | 852.96 | 995.28 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=26.09 |
| Stop hit — per-position SL triggered | 2026-02-23 05:30:00 | 982.91 | 857.51 | 997.92 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 967.80 | 871.58 | 906.36 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=37.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 05:30:00 | 1043.68 | 882.29 | 962.84 | T1 booked 50% @ 1043.68 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-23 05:30:00 | 731.65 | 2025-10-06 05:30:00 | 763.61 | PARTIAL | 0.50 | 4.37% |
| BUY | retest1 | 2025-09-23 05:30:00 | 731.65 | 2026-01-13 05:30:00 | 971.95 | TARGET_HIT | 0.50 | 32.84% |
| BUY | retest1 | 2026-01-16 05:30:00 | 1025.40 | 2026-01-21 05:30:00 | 991.24 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2026-02-18 05:30:00 | 1022.05 | 2026-02-23 05:30:00 | 982.91 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest1 | 2026-04-08 05:30:00 | 967.80 | 2026-04-22 05:30:00 | 1043.68 | PARTIAL | 0.50 | 7.84% |
