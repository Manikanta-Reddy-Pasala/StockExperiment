# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 328.20
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 2.54% / 2.74%
- **Sum % (uncompounded):** 12.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.54% | 12.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.54% | 12.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 2.54% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 05:30:00 | 251.89 | 247.24 | 243.23 | Stage2 pullback-breakout RSI=59 vol=2.5x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 05:30:00 | 266.21 | 247.80 | 249.92 | T1 booked 50% @ 266.21 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 258.80 | 248.60 | 255.77 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 05:30:00 | 264.40 | 249.16 | 253.62 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=10.06 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 249.30 | 249.22 | 253.36 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2026-03-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 05:30:00 | 265.25 | 251.22 | 255.48 | Stage2 pullback-breakout RSI=57 vol=6.0x ATR=11.12 |
| Stop hit — per-position SL triggered | 2026-03-27 05:30:00 | 270.40 | 252.22 | 259.87 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2026-04-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 05:30:00 | 297.75 | 254.62 | 270.70 | Stage2 pullback-breakout RSI=69 vol=9.7x ATR=11.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 05:30:00 | 321.69 | 257.84 | 286.83 | T1 booked 50% @ 321.69 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-23 05:30:00 | 251.89 | 2026-01-02 05:30:00 | 266.21 | PARTIAL | 0.50 | 5.69% |
| BUY | retest1 | 2025-12-23 05:30:00 | 251.89 | 2026-01-08 05:30:00 | 258.80 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest1 | 2026-01-29 05:30:00 | 264.40 | 2026-02-01 05:30:00 | 249.30 | STOP_HIT | 1.00 | -5.71% |
| BUY | retest1 | 2026-03-12 05:30:00 | 265.25 | 2026-03-27 05:30:00 | 270.40 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest1 | 2026-04-16 05:30:00 | 297.75 | 2026-04-27 05:30:00 | 321.69 | PARTIAL | 0.50 | 8.04% |
