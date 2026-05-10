# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1507.30
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.05% / 5.75%
- **Sum % (uncompounded):** 15.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.05% | 15.3% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.05% | 15.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.05% | 15.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 1696.50 | 1518.22 | 1646.03 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=49.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 05:30:00 | 1795.40 | 1528.54 | 1680.26 | T1 booked 50% @ 1795.40 |
| Stop hit — per-position SL triggered | 2025-07-24 05:30:00 | 1696.50 | 1540.14 | 1712.47 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2025-10-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 05:30:00 | 1582.90 | 1558.65 | 1557.60 | Stage2 pullback-breakout RSI=53 vol=5.1x ATR=45.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 05:30:00 | 1673.93 | 1561.41 | 1581.80 | T1 booked 50% @ 1673.93 |
| Target hit | 2025-11-12 05:30:00 | 1701.20 | 1590.91 | 1715.36 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 1667.10 | 1608.13 | 1624.80 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=42.07 |
| Stop hit — per-position SL triggered | 2026-01-07 05:30:00 | 1604.00 | 1608.69 | 1626.74 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 1696.50 | 2025-07-17 05:30:00 | 1795.40 | PARTIAL | 0.50 | 5.83% |
| BUY | retest1 | 2025-07-10 05:30:00 | 1696.50 | 2025-07-24 05:30:00 | 1696.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-09 05:30:00 | 1582.90 | 2025-10-15 05:30:00 | 1673.93 | PARTIAL | 0.50 | 5.75% |
| BUY | retest1 | 2025-10-09 05:30:00 | 1582.90 | 2025-11-12 05:30:00 | 1701.20 | TARGET_HIT | 0.50 | 7.47% |
| BUY | retest1 | 2026-01-05 05:30:00 | 1667.10 | 2026-01-07 05:30:00 | 1604.00 | STOP_HIT | 1.00 | -3.78% |
