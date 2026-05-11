# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1476.00
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.62% / -2.87%
- **Sum % (uncompounded):** -7.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.62% | -7.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.62% | -7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.62% | -7.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 05:30:00 | 1580.83 | 1412.76 | 1554.86 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=29.51 |
| Stop hit — per-position SL triggered | 2024-07-25 05:30:00 | 1554.78 | 1428.31 | 1567.03 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 05:30:00 | 1681.73 | 1486.67 | 1624.38 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=32.23 |
| Stop hit — per-position SL triggered | 2024-10-03 05:30:00 | 1633.39 | 1491.98 | 1635.06 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 05:30:00 | 1602.38 | 1512.60 | 1541.75 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=35.57 |
| Stop hit — per-position SL triggered | 2024-12-17 05:30:00 | 1549.02 | 1519.01 | 1567.51 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-10 05:30:00 | 1580.83 | 2024-07-25 05:30:00 | 1554.78 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2024-09-27 05:30:00 | 1681.73 | 2024-10-03 05:30:00 | 1633.39 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest1 | 2024-12-04 05:30:00 | 1602.38 | 2024-12-17 05:30:00 | 1549.02 | STOP_HIT | 1.00 | -3.33% |
