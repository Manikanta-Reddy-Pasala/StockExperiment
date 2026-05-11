# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2025-09-03 05:30:00 (497 bars)
- **Last close:** 1352.00
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
- **Avg / median % per leg:** 0.15% / 4.82%
- **Sum % (uncompounded):** 0.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.15% | 0.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.15% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.15% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 05:30:00 | 1480.00 | 1234.59 | 1431.59 | Stage2 pullback-breakout RSI=65 vol=3.3x ATR=35.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 05:30:00 | 1551.36 | 1260.33 | 1482.68 | T1 booked 50% @ 1551.36 |
| Target hit | 2024-10-23 05:30:00 | 1565.80 | 1295.95 | 1574.74 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 1607.15 | 1348.32 | 1571.68 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=46.68 |
| Stop hit — per-position SL triggered | 2024-11-27 05:30:00 | 1537.13 | 1352.26 | 1567.10 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-03-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 05:30:00 | 1501.90 | 1425.46 | 1409.74 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=56.62 |
| Stop hit — per-position SL triggered | 2025-04-01 05:30:00 | 1416.97 | 1425.12 | 1407.94 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-24 05:30:00 | 1480.00 | 2024-10-09 05:30:00 | 1551.36 | PARTIAL | 0.50 | 4.82% |
| BUY | retest1 | 2024-09-24 05:30:00 | 1480.00 | 2024-10-23 05:30:00 | 1565.80 | TARGET_HIT | 0.50 | 5.80% |
| BUY | retest1 | 2024-11-25 05:30:00 | 1607.15 | 2024-11-27 05:30:00 | 1537.13 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest1 | 2025-03-28 05:30:00 | 1501.90 | 2025-04-01 05:30:00 | 1416.97 | STOP_HIT | 1.00 | -5.65% |
