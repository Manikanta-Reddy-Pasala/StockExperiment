# Gujarat State Petronet Ltd. (GSPL)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 288.95
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 4.09% / 4.54%
- **Sum % (uncompounded):** 24.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.09% | 24.5% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.09% | 24.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.09% | 24.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 05:30:00 | 288.50 | 276.91 | 278.29 | Stage2 pullback-breakout RSI=63 vol=1.5x ATR=6.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 05:30:00 | 301.61 | 277.58 | 283.01 | T1 booked 50% @ 301.61 |
| Stop hit — per-position SL triggered | 2023-12-14 05:30:00 | 293.50 | 278.32 | 287.31 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-12-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 05:30:00 | 308.60 | 279.53 | 291.19 | Stage2 pullback-breakout RSI=67 vol=3.3x ATR=8.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 05:30:00 | 326.29 | 280.69 | 298.54 | T1 booked 50% @ 326.29 |
| Stop hit — per-position SL triggered | 2024-01-11 05:30:00 | 316.35 | 283.76 | 311.98 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-04-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 05:30:00 | 367.00 | 314.35 | 354.33 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=13.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 05:30:00 | 393.09 | 320.34 | 370.09 | T1 booked 50% @ 393.09 |
| Stop hit — per-position SL triggered | 2024-04-19 05:30:00 | 377.65 | 321.56 | 372.17 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-30 05:30:00 | 288.50 | 2023-12-08 05:30:00 | 301.61 | PARTIAL | 0.50 | 4.54% |
| BUY | retest1 | 2023-11-30 05:30:00 | 288.50 | 2023-12-14 05:30:00 | 293.50 | STOP_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2023-12-28 05:30:00 | 308.60 | 2024-01-02 05:30:00 | 326.29 | PARTIAL | 0.50 | 5.73% |
| BUY | retest1 | 2023-12-28 05:30:00 | 308.60 | 2024-01-11 05:30:00 | 316.35 | STOP_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2024-04-01 05:30:00 | 367.00 | 2024-04-16 05:30:00 | 393.09 | PARTIAL | 0.50 | 7.11% |
| BUY | retest1 | 2024-04-01 05:30:00 | 367.00 | 2024-04-19 05:30:00 | 377.65 | STOP_HIT | 0.50 | 2.90% |
