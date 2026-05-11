# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 146.61
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.44% / 1.21%
- **Sum % (uncompounded):** 12.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.44% | 12.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.44% | 12.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 2.44% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 00:00:00 | 125.02 | 93.95 | 113.73 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=6.52 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 115.24 | 96.72 | 118.90 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 129.82 | 98.09 | 119.82 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=6.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 00:00:00 | 143.12 | 101.43 | 128.53 | T1 booked 50% @ 143.12 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 131.39 | 102.07 | 129.48 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 00:00:00 | 157.19 | 112.09 | 149.17 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 00:00:00 | 170.69 | 113.49 | 151.85 | T1 booked 50% @ 170.69 |
| Stop hit — per-position SL triggered | 2024-09-24 00:00:00 | 157.19 | 113.92 | 152.29 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-04 00:00:00 | 125.02 | 2024-07-19 00:00:00 | 115.24 | STOP_HIT | 1.00 | -7.82% |
| BUY | retest1 | 2024-07-29 00:00:00 | 129.82 | 2024-08-12 00:00:00 | 143.12 | PARTIAL | 0.50 | 10.24% |
| BUY | retest1 | 2024-07-29 00:00:00 | 129.82 | 2024-08-14 00:00:00 | 131.39 | STOP_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2024-09-18 00:00:00 | 157.19 | 2024-09-23 00:00:00 | 170.69 | PARTIAL | 0.50 | 8.59% |
| BUY | retest1 | 2024-09-18 00:00:00 | 157.19 | 2024-09-24 00:00:00 | 157.19 | STOP_HIT | 0.50 | 0.00% |
