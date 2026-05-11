# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1639.90
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 3.66% / 6.46%
- **Sum % (uncompounded):** 21.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.66% | 21.9% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.66% | 21.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 3 | 3 | 3.66% | 21.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 00:00:00 | 1763.60 | 1297.97 | 1625.00 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=68.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 00:00:00 | 1901.22 | 1311.84 | 1662.61 | T1 booked 50% @ 1901.22 |
| Stop hit — per-position SL triggered | 2024-07-15 00:00:00 | 1763.60 | 1331.00 | 1707.93 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-09-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 00:00:00 | 1793.00 | 1465.05 | 1722.53 | Stage2 pullback-breakout RSI=61 vol=3.5x ATR=57.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 00:00:00 | 1908.89 | 1486.12 | 1771.31 | T1 booked 50% @ 1908.89 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 1793.00 | 1489.75 | 1778.88 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-11-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 00:00:00 | 1978.10 | 1593.57 | 1879.33 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=75.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 00:00:00 | 2129.87 | 1603.54 | 1919.20 | T1 booked 50% @ 2129.87 |
| Stop hit — per-position SL triggered | 2024-11-14 00:00:00 | 1978.10 | 1632.96 | 1998.61 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-04 00:00:00 | 1763.60 | 2024-07-09 00:00:00 | 1901.22 | PARTIAL | 0.50 | 7.80% |
| BUY | retest1 | 2024-07-04 00:00:00 | 1763.60 | 2024-07-15 00:00:00 | 1763.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 00:00:00 | 1793.00 | 2024-09-18 00:00:00 | 1908.89 | PARTIAL | 0.50 | 6.46% |
| BUY | retest1 | 2024-09-10 00:00:00 | 1793.00 | 2024-09-19 00:00:00 | 1793.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-04 00:00:00 | 1978.10 | 2024-11-06 00:00:00 | 2129.87 | PARTIAL | 0.50 | 7.67% |
| BUY | retest1 | 2024-11-04 00:00:00 | 1978.10 | 2024-11-14 00:00:00 | 1978.10 | STOP_HIT | 0.50 | 0.00% |
