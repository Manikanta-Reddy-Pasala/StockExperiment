# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1948.60
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
- **Avg / median % per leg:** 1.67% / 4.73%
- **Sum % (uncompounded):** 6.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.67% | 6.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.67% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.67% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 05:30:00 | 1731.10 | 1626.66 | 1682.58 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=34.88 |
| Stop hit — per-position SL triggered | 2025-10-16 05:30:00 | 1728.00 | 1636.91 | 1714.90 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-01-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 05:30:00 | 1976.00 | 1726.12 | 1902.24 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=46.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 05:30:00 | 2069.48 | 1728.85 | 1911.56 | T1 booked 50% @ 2069.48 |
| Target hit | 2026-03-04 05:30:00 | 2119.70 | 1836.22 | 2180.93 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-04-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 05:30:00 | 2143.90 | 1878.67 | 2027.51 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=73.48 |
| Stop hit — per-position SL triggered | 2026-04-23 05:30:00 | 2033.67 | 1884.72 | 2041.37 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-01 05:30:00 | 1731.10 | 2025-10-16 05:30:00 | 1728.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-01-22 05:30:00 | 1976.00 | 2026-01-23 05:30:00 | 2069.48 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2026-01-22 05:30:00 | 1976.00 | 2026-03-04 05:30:00 | 2119.70 | TARGET_HIT | 0.50 | 7.27% |
| BUY | retest1 | 2026-04-20 05:30:00 | 2143.90 | 2026-04-23 05:30:00 | 2033.67 | STOP_HIT | 1.00 | -5.14% |
