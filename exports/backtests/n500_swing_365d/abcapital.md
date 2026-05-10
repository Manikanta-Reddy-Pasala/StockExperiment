# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 362.95
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** 0.34% / 1.80%
- **Sum % (uncompounded):** 2.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 5 | 1 | 0.34% | 2.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 5 | 1 | 0.34% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 5 | 1 | 0.34% | 2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 05:30:00 | 278.45 | 220.47 | 263.08 | Stage2 pullback-breakout RSI=61 vol=6.7x ATR=8.93 |
| Stop hit — per-position SL triggered | 2025-08-20 05:30:00 | 283.45 | 226.35 | 272.77 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 05:30:00 | 345.45 | 264.85 | 327.38 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=8.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 05:30:00 | 362.22 | 273.45 | 345.37 | T1 booked 50% @ 362.22 |
| Stop hit — per-position SL triggered | 2025-12-11 05:30:00 | 354.45 | 275.10 | 347.41 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 357.70 | 284.17 | 348.96 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=7.94 |
| Stop hit — per-position SL triggered | 2026-01-13 05:30:00 | 345.79 | 290.57 | 354.07 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 353.10 | 305.29 | 347.30 | Stage2 pullback-breakout RSI=55 vol=2.5x ATR=10.73 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 337.01 | 306.50 | 346.87 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 338.40 | 308.14 | 315.68 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=14.17 |
| Stop hit — per-position SL triggered | 2026-04-24 05:30:00 | 340.60 | 311.68 | 333.93 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2026-05-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 05:30:00 | 360.85 | 313.74 | 340.23 | Stage2 pullback-breakout RSI=64 vol=2.9x ATR=12.65 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-04 05:30:00 | 278.45 | 2025-08-20 05:30:00 | 283.45 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest1 | 2025-11-25 05:30:00 | 345.45 | 2025-12-09 05:30:00 | 362.22 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2025-11-25 05:30:00 | 345.45 | 2025-12-11 05:30:00 | 354.45 | STOP_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2025-12-31 05:30:00 | 357.70 | 2026-01-13 05:30:00 | 345.79 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2026-02-25 05:30:00 | 353.10 | 2026-03-02 05:30:00 | 337.01 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest1 | 2026-04-08 05:30:00 | 338.40 | 2026-04-24 05:30:00 | 340.60 | STOP_HIT | 1.00 | 0.65% |
