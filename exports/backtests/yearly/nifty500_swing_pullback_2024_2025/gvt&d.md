# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (662 bars)
- **Last close:** 4625.50
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
- **Avg / median % per leg:** -1.15% / 0.00%
- **Sum % (uncompounded):** -4.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.15% | -4.6% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.15% | -4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.15% | -4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 05:30:00 | 1664.30 | 1027.99 | 1574.49 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=86.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 05:30:00 | 1837.13 | 1108.00 | 1691.31 | T1 booked 50% @ 1837.13 |
| Stop hit — per-position SL triggered | 2024-08-22 05:30:00 | 1664.30 | 1119.27 | 1688.44 | SL hit (bars_held=14) |

### Cycle 2 — BUY (started 2024-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 05:30:00 | 1866.00 | 1374.13 | 1753.96 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=94.99 |
| Stop hit — per-position SL triggered | 2024-11-18 05:30:00 | 1723.52 | 1386.67 | 1766.10 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-03-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 05:30:00 | 1622.05 | 1569.85 | 1489.68 | Stage2 pullback-breakout RSI=62 vol=2.2x ATR=79.29 |
| Stop hit — per-position SL triggered | 2025-04-01 05:30:00 | 1503.12 | 1569.38 | 1520.15 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-01 05:30:00 | 1664.30 | 2024-08-20 05:30:00 | 1837.13 | PARTIAL | 0.50 | 10.38% |
| BUY | retest1 | 2024-08-01 05:30:00 | 1664.30 | 2024-08-22 05:30:00 | 1664.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 05:30:00 | 1866.00 | 2024-11-18 05:30:00 | 1723.52 | STOP_HIT | 1.00 | -7.64% |
| BUY | retest1 | 2025-03-21 05:30:00 | 1622.05 | 2025-04-01 05:30:00 | 1503.12 | STOP_HIT | 1.00 | -7.33% |
