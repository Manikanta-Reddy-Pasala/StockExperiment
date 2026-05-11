# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1927.60
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
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** -0.81% / 0.76%
- **Sum % (uncompounded):** -3.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.81% | -3.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.81% | -3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.81% | -3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 2432.20 | 2009.22 | 2335.33 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=99.88 |
| Stop hit — per-position SL triggered | 2025-07-02 05:30:00 | 2282.39 | 2019.80 | 2341.80 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-07-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 05:30:00 | 2428.10 | 2064.19 | 2338.10 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=87.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 05:30:00 | 2603.10 | 2081.35 | 2394.83 | T1 booked 50% @ 2603.10 |
| Target hit | 2025-08-07 05:30:00 | 2446.60 | 2109.49 | 2466.68 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 2358.50 | 2203.60 | 2297.12 | Stage2 pullback-breakout RSI=58 vol=2.5x ATR=79.53 |
| Stop hit — per-position SL triggered | 2025-12-30 05:30:00 | 2239.21 | 2212.65 | 2321.63 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 2432.20 | 2025-07-02 05:30:00 | 2282.39 | STOP_HIT | 1.00 | -6.16% |
| BUY | retest1 | 2025-07-24 05:30:00 | 2428.10 | 2025-07-30 05:30:00 | 2603.10 | PARTIAL | 0.50 | 7.21% |
| BUY | retest1 | 2025-07-24 05:30:00 | 2428.10 | 2025-08-07 05:30:00 | 2446.60 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2025-12-19 05:30:00 | 2358.50 | 2025-12-30 05:30:00 | 2239.21 | STOP_HIT | 1.00 | -5.06% |
