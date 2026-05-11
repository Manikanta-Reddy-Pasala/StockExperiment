# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1803.20
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
- **Avg / median % per leg:** -0.32% / 2.65%
- **Sum % (uncompounded):** -1.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 1884.70 | 1513.46 | 1789.89 | Stage2 pullback-breakout RSI=60 vol=3.2x ATR=73.02 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 1775.17 | 1530.49 | 1795.04 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 00:00:00 | 1844.05 | 1548.97 | 1769.18 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=87.32 |
| Stop hit — per-position SL triggered | 2024-09-27 00:00:00 | 1713.07 | 1562.08 | 1789.11 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 1670.75 | 1564.13 | 1537.93 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=75.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 1821.04 | 1582.66 | 1685.61 | T1 booked 50% @ 1821.04 |
| Target hit | 2024-12-18 00:00:00 | 1714.95 | 1595.27 | 1733.74 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-30 00:00:00 | 1884.70 | 2024-09-09 00:00:00 | 1775.17 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest1 | 2024-09-20 00:00:00 | 1844.05 | 2024-09-27 00:00:00 | 1713.07 | STOP_HIT | 1.00 | -7.10% |
| BUY | retest1 | 2024-11-25 00:00:00 | 1670.75 | 2024-12-10 00:00:00 | 1821.04 | PARTIAL | 0.50 | 9.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 1670.75 | 2024-12-18 00:00:00 | 1714.95 | TARGET_HIT | 0.50 | 2.65% |
