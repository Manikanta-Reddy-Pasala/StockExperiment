# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -1.17% / 0.00%
- **Sum % (uncompounded):** -4.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.17% | -4.7% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.17% | -4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.17% | -4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 05:30:00 | 1777.95 | 1478.53 | 1658.30 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=53.87 |
| Stop hit — per-position SL triggered | 2024-11-13 05:30:00 | 1697.15 | 1488.23 | 1679.83 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-12-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 05:30:00 | 1819.25 | 1541.81 | 1764.68 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=44.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 05:30:00 | 1908.10 | 1574.65 | 1830.98 | T1 booked 50% @ 1908.10 |
| Stop hit — per-position SL triggered | 2025-01-13 05:30:00 | 1819.25 | 1601.74 | 1879.88 | SL hit (bars_held=19) |

### Cycle 3 — BUY (started 2025-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 05:30:00 | 1878.15 | 1635.41 | 1825.82 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=63.05 |
| Stop hit — per-position SL triggered | 2025-02-12 05:30:00 | 1783.57 | 1648.08 | 1836.89 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-11-07 05:30:00 | 1777.95 | 2024-11-13 05:30:00 | 1697.15 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-12-16 05:30:00 | 1819.25 | 2025-01-01 05:30:00 | 1908.10 | PARTIAL | 0.50 | 4.88% |
| BUY | retest1 | 2024-12-16 05:30:00 | 1819.25 | 2025-01-13 05:30:00 | 1819.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-04 05:30:00 | 1878.15 | 2025-02-12 05:30:00 | 1783.57 | STOP_HIT | 1.00 | -5.04% |
