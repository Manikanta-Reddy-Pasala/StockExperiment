# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1942.40
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 1.50% / 2.21%
- **Sum % (uncompounded):** 4.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.50% | 4.5% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.50% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.50% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 1487.00 | 1321.33 | 1410.80 | Stage2 pullback-breakout RSI=69 vol=3.8x ATR=42.05 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 1519.80 | 1342.90 | 1478.05 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-02-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 05:30:00 | 1502.00 | 1365.90 | 1434.79 | Stage2 pullback-breakout RSI=63 vol=3.8x ATR=51.45 |
| Stop hit — per-position SL triggered | 2026-03-11 05:30:00 | 1424.82 | 1378.68 | 1471.45 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2026-04-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 05:30:00 | 1455.00 | 1376.68 | 1384.54 | Stage2 pullback-breakout RSI=59 vol=2.9x ATR=54.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 05:30:00 | 1563.01 | 1384.85 | 1442.43 | T1 booked 50% @ 1563.01 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-19 05:30:00 | 1487.00 | 2026-01-08 05:30:00 | 1519.80 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest1 | 2026-02-24 05:30:00 | 1502.00 | 2026-03-11 05:30:00 | 1424.82 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest1 | 2026-04-16 05:30:00 | 1455.00 | 2026-04-28 05:30:00 | 1563.01 | PARTIAL | 0.50 | 7.42% |
