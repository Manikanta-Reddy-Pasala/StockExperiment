# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 739.90
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
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 7.54% / 7.67%
- **Sum % (uncompounded):** 37.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.54% | 37.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.54% | 37.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.54% | 37.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 00:00:00 | 462.00 | 395.19 | 441.58 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=17.36 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 435.96 | 400.00 | 446.54 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-10-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 00:00:00 | 522.35 | 421.99 | 482.75 | Stage2 pullback-breakout RSI=68 vol=4.0x ATR=20.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 00:00:00 | 563.87 | 429.70 | 510.50 | T1 booked 50% @ 563.87 |
| Target hit | 2023-11-17 00:00:00 | 562.40 | 458.48 | 563.22 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 567.95 | 494.76 | 552.55 | Stage2 pullback-breakout RSI=59 vol=3.4x ATR=18.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 00:00:00 | 604.76 | 496.63 | 559.26 | T1 booked 50% @ 604.76 |
| Target hit | 2024-03-06 00:00:00 | 688.60 | 540.54 | 695.66 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 690.75 | 567.03 | 649.43 | Stage2 pullback-breakout RSI=64 vol=7.0x ATR=31.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-01 00:00:00 | 462.00 | 2023-08-14 00:00:00 | 435.96 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest1 | 2023-10-05 00:00:00 | 522.35 | 2023-10-16 00:00:00 | 563.87 | PARTIAL | 0.50 | 7.95% |
| BUY | retest1 | 2023-10-05 00:00:00 | 522.35 | 2023-11-17 00:00:00 | 562.40 | TARGET_HIT | 0.50 | 7.67% |
| BUY | retest1 | 2024-01-25 00:00:00 | 567.95 | 2024-01-30 00:00:00 | 604.76 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2024-01-25 00:00:00 | 567.95 | 2024-03-06 00:00:00 | 688.60 | TARGET_HIT | 0.50 | 21.24% |
