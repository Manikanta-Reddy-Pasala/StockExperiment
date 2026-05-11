# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 409.25
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
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -1.16% / -3.38%
- **Sum % (uncompounded):** -8.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.16% | -8.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.16% | -8.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | -1.16% | -8.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 05:30:00 | 418.15 | 336.72 | 403.96 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=10.07 |
| Stop hit — per-position SL triggered | 2023-07-21 05:30:00 | 418.55 | 344.92 | 415.20 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 05:30:00 | 389.90 | 365.32 | 378.85 | Stage2 pullback-breakout RSI=58 vol=3.3x ATR=8.78 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 376.73 | 365.96 | 379.68 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 05:30:00 | 410.30 | 367.80 | 384.02 | Stage2 pullback-breakout RSI=67 vol=6.0x ATR=10.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 05:30:00 | 430.39 | 370.91 | 401.11 | T1 booked 50% @ 430.39 |
| Stop hit — per-position SL triggered | 2023-11-22 05:30:00 | 420.85 | 373.05 | 409.05 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 05:30:00 | 542.00 | 434.39 | 524.23 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=13.47 |
| Stop hit — per-position SL triggered | 2024-03-07 05:30:00 | 521.80 | 436.20 | 524.47 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-04-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 05:30:00 | 488.75 | 443.21 | 477.82 | Stage2 pullback-breakout RSI=54 vol=2.2x ATR=13.65 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 468.27 | 443.90 | 477.78 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 05:30:00 | 510.10 | 448.06 | 485.73 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=14.96 |
| Stop hit — per-position SL triggered | 2024-05-06 05:30:00 | 487.66 | 449.58 | 489.05 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-07 05:30:00 | 418.15 | 2023-07-21 05:30:00 | 418.55 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest1 | 2023-10-17 05:30:00 | 389.90 | 2023-10-23 05:30:00 | 376.73 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest1 | 2023-11-08 05:30:00 | 410.30 | 2023-11-16 05:30:00 | 430.39 | PARTIAL | 0.50 | 4.90% |
| BUY | retest1 | 2023-11-08 05:30:00 | 410.30 | 2023-11-22 05:30:00 | 420.85 | STOP_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2024-03-05 05:30:00 | 542.00 | 2024-03-07 05:30:00 | 521.80 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2024-04-10 05:30:00 | 488.75 | 2024-04-15 05:30:00 | 468.27 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2024-04-30 05:30:00 | 510.10 | 2024-05-06 05:30:00 | 487.66 | STOP_HIT | 1.00 | -4.40% |
