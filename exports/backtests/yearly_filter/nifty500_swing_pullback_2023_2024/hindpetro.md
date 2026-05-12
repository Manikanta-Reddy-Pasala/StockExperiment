# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 378.55
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 10.64% / 5.23%
- **Sum % (uncompounded):** 63.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 10.64% | 63.8% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 10.64% | 63.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 10.64% | 63.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 189.00 | 162.97 | 180.28 | Stage2 pullback-breakout RSI=66 vol=3.6x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 199.37 | 163.88 | 183.90 | T1 booked 50% @ 199.37 |
| Stop hit — per-position SL triggered | 2023-07-14 00:00:00 | 189.00 | 165.76 | 189.08 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 174.73 | 170.13 | 169.00 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 00:00:00 | 183.86 | 170.28 | 170.60 | T1 booked 50% @ 183.86 |
| Target hit | 2024-01-25 00:00:00 | 287.07 | 207.74 | 289.08 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 324.97 | 257.40 | 318.05 | Stage2 pullback-breakout RSI=53 vol=1.8x ATR=12.24 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 306.62 | 258.55 | 317.51 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-05-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 00:00:00 | 355.67 | 265.76 | 326.17 | Stage2 pullback-breakout RSI=67 vol=2.7x ATR=13.11 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 336.00 | 269.61 | 333.22 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 189.00 | 2023-07-06 00:00:00 | 199.37 | PARTIAL | 0.50 | 5.49% |
| BUY | retest1 | 2023-07-03 00:00:00 | 189.00 | 2023-07-14 00:00:00 | 189.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 00:00:00 | 174.73 | 2023-11-07 00:00:00 | 183.86 | PARTIAL | 0.50 | 5.23% |
| BUY | retest1 | 2023-11-06 00:00:00 | 174.73 | 2024-01-25 00:00:00 | 287.07 | TARGET_HIT | 0.50 | 64.29% |
| BUY | retest1 | 2024-04-10 00:00:00 | 324.97 | 2024-04-15 00:00:00 | 306.62 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2024-05-02 00:00:00 | 355.67 | 2024-05-09 00:00:00 | 336.00 | STOP_HIT | 1.00 | -5.53% |
