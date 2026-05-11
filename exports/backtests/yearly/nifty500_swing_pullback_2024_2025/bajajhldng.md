# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 10632.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.33% / -0.06%
- **Sum % (uncompounded):** -1.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 05:30:00 | 10793.00 | 8817.24 | 10318.46 | Stage2 pullback-breakout RSI=66 vol=2.0x ATR=295.56 |
| Stop hit — per-position SL triggered | 2024-09-30 05:30:00 | 10349.66 | 8969.37 | 10562.35 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 05:30:00 | 10658.25 | 9275.96 | 10382.59 | Stage2 pullback-breakout RSI=59 vol=3.4x ATR=305.87 |
| Stop hit — per-position SL triggered | 2024-11-19 05:30:00 | 10555.30 | 9406.27 | 10553.48 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-12-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 05:30:00 | 11122.00 | 9546.83 | 10598.49 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=336.47 |
| Stop hit — per-position SL triggered | 2024-12-23 05:30:00 | 11115.15 | 9698.70 | 10933.61 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-01-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 05:30:00 | 11286.40 | 9980.61 | 11033.43 | Stage2 pullback-breakout RSI=54 vol=3.6x ATR=455.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 05:30:00 | 12197.66 | 10105.90 | 11312.70 | T1 booked 50% @ 12197.66 |
| Stop hit — per-position SL triggered | 2025-02-05 05:30:00 | 11381.90 | 10134.65 | 11355.75 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-03-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 05:30:00 | 12273.55 | 10547.66 | 11710.85 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=470.21 |
| Stop hit — per-position SL triggered | 2025-04-02 05:30:00 | 11568.24 | 10663.98 | 11963.28 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-18 05:30:00 | 10793.00 | 2024-09-30 05:30:00 | 10349.66 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2024-11-04 05:30:00 | 10658.25 | 2024-11-19 05:30:00 | 10555.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-12-09 05:30:00 | 11122.00 | 2024-12-23 05:30:00 | 11115.15 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest1 | 2025-01-22 05:30:00 | 11286.40 | 2025-02-03 05:30:00 | 12197.66 | PARTIAL | 0.50 | 8.07% |
| BUY | retest1 | 2025-01-22 05:30:00 | 11286.40 | 2025-02-05 05:30:00 | 11381.90 | STOP_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2025-03-21 05:30:00 | 12273.55 | 2025-04-02 05:30:00 | 11568.24 | STOP_HIT | 1.00 | -5.75% |
