# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2025-09-03 05:30:00 (497 bars)
- **Last close:** 1174.00
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
- **Avg / median % per leg:** -0.22% / 0.00%
- **Sum % (uncompounded):** -0.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.22% | -0.9% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.22% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.22% | -0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 05:30:00 | 1519.50 | 1402.70 | 1467.73 | Stage2 pullback-breakout RSI=63 vol=3.5x ATR=52.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 05:30:00 | 1623.65 | 1409.00 | 1500.51 | T1 booked 50% @ 1623.65 |
| Stop hit — per-position SL triggered | 2024-08-02 05:30:00 | 1519.50 | 1412.90 | 1520.03 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 05:30:00 | 1578.95 | 1443.24 | 1529.00 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=50.34 |
| Stop hit — per-position SL triggered | 2024-09-24 05:30:00 | 1521.55 | 1454.26 | 1545.33 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-10-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 05:30:00 | 1626.90 | 1459.71 | 1555.02 | Stage2 pullback-breakout RSI=64 vol=3.1x ATR=44.46 |
| Stop hit — per-position SL triggered | 2024-10-04 05:30:00 | 1560.21 | 1462.22 | 1560.64 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-25 05:30:00 | 1519.50 | 2024-07-31 05:30:00 | 1623.65 | PARTIAL | 0.50 | 6.85% |
| BUY | retest1 | 2024-07-25 05:30:00 | 1519.50 | 2024-08-02 05:30:00 | 1519.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 05:30:00 | 1578.95 | 2024-09-24 05:30:00 | 1521.55 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest1 | 2024-10-01 05:30:00 | 1626.90 | 2024-10-04 05:30:00 | 1560.21 | STOP_HIT | 1.00 | -4.10% |
