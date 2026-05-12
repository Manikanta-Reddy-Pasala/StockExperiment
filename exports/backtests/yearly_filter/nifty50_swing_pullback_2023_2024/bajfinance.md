# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 1.35% / 2.76%
- **Sum % (uncompounded):** 5.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.35% | 5.4% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.35% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.35% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 733.30 | 663.47 | 708.43 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=11.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 756.79 | 664.69 | 715.82 | T1 booked 50% @ 756.79 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 751.10 | 672.58 | 738.57 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 00:00:00 | 721.15 | 684.60 | 714.16 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=13.99 |
| Stop hit — per-position SL triggered | 2023-09-08 00:00:00 | 741.07 | 688.99 | 725.56 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 770.56 | 721.29 | 738.38 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=15.47 |
| Stop hit — per-position SL triggered | 2024-01-15 00:00:00 | 747.35 | 724.34 | 751.91 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 733.30 | 2023-07-04 00:00:00 | 756.79 | PARTIAL | 0.50 | 3.20% |
| BUY | retest1 | 2023-07-03 00:00:00 | 733.30 | 2023-07-17 00:00:00 | 751.10 | STOP_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2023-08-25 00:00:00 | 721.15 | 2023-09-08 00:00:00 | 741.07 | STOP_HIT | 1.00 | 2.76% |
| BUY | retest1 | 2024-01-04 00:00:00 | 770.56 | 2024-01-15 00:00:00 | 747.35 | STOP_HIT | 1.00 | -3.01% |
