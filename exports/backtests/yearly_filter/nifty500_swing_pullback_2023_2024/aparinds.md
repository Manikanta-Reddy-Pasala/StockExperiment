# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 12810.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.82% / 2.60%
- **Sum % (uncompounded):** 14.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.82% | 14.1% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.82% | 14.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.82% | 14.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 5428.40 | 3807.09 | 5189.91 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=210.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 00:00:00 | 5849.39 | 3861.86 | 5313.36 | T1 booked 50% @ 5849.39 |
| Stop hit — per-position SL triggered | 2023-11-28 00:00:00 | 5428.40 | 3966.64 | 5461.77 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 5657.65 | 4206.06 | 5423.71 | Stage2 pullback-breakout RSI=62 vol=2.6x ATR=176.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 00:00:00 | 6011.45 | 4253.32 | 5525.53 | T1 booked 50% @ 6011.45 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 5804.85 | 4382.36 | 5740.67 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 6068.75 | 4530.95 | 5607.35 | Stage2 pullback-breakout RSI=64 vol=5.5x ATR=244.10 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 5915.10 | 4689.59 | 5961.68 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-13 00:00:00 | 5428.40 | 2023-11-17 00:00:00 | 5849.39 | PARTIAL | 0.50 | 7.76% |
| BUY | retest1 | 2023-11-13 00:00:00 | 5428.40 | 2023-11-28 00:00:00 | 5428.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 00:00:00 | 5657.65 | 2023-12-28 00:00:00 | 6011.45 | PARTIAL | 0.50 | 6.25% |
| BUY | retest1 | 2023-12-22 00:00:00 | 5657.65 | 2024-01-09 00:00:00 | 5804.85 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2024-01-30 00:00:00 | 6068.75 | 2024-02-13 00:00:00 | 5915.10 | STOP_HIT | 1.00 | -2.53% |
