# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1636.50
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
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 7.78% / 9.84%
- **Sum % (uncompounded):** 23.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.78% | 23.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.78% | 23.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.78% | 23.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 00:00:00 | 781.00 | 590.42 | 730.82 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=31.96 |
| Stop hit — per-position SL triggered | 2023-08-09 00:00:00 | 733.06 | 595.66 | 739.93 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-02-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 00:00:00 | 1000.85 | 759.21 | 926.88 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=49.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 00:00:00 | 1099.36 | 772.15 | 967.12 | T1 booked 50% @ 1099.36 |
| Target hit | 2024-04-19 00:00:00 | 1197.45 | 889.07 | 1202.64 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-04 00:00:00 | 781.00 | 2023-08-09 00:00:00 | 733.06 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest1 | 2024-02-19 00:00:00 | 1000.85 | 2024-02-26 00:00:00 | 1099.36 | PARTIAL | 0.50 | 9.84% |
| BUY | retest1 | 2024-02-19 00:00:00 | 1000.85 | 2024-04-19 00:00:00 | 1197.45 | TARGET_HIT | 0.50 | 19.64% |
