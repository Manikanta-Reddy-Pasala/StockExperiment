# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 168.57
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.50% / 0.88%
- **Sum % (uncompounded):** 10.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.50% | 10.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.50% | 10.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.50% | 10.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 84.90 | 75.87 | 81.55 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 00:00:00 | 89.02 | 76.63 | 84.48 | T1 booked 50% @ 89.02 |
| Target hit | 2023-08-31 00:00:00 | 91.93 | 80.52 | 92.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 89.03 | 83.48 | 86.54 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=1.87 |
| Stop hit — per-position SL triggered | 2023-12-06 00:00:00 | 88.35 | 84.00 | 88.01 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 90.78 | 84.48 | 87.47 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-01-10 00:00:00 | 87.63 | 84.89 | 88.57 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 89.63 | 85.22 | 87.35 | Stage2 pullback-breakout RSI=58 vol=4.0x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 86.07 | 85.34 | 87.65 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 87.33 | 85.18 | 84.25 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 88.10 | 85.45 | 86.70 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 92.50 | 85.60 | 87.41 | Stage2 pullback-breakout RSI=67 vol=2.9x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 96.86 | 85.78 | 88.71 | T1 booked 50% @ 96.86 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 84.90 | 2023-07-21 00:00:00 | 89.02 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2023-07-12 00:00:00 | 84.90 | 2023-08-31 00:00:00 | 91.93 | TARGET_HIT | 0.50 | 8.28% |
| BUY | retest1 | 2023-11-21 00:00:00 | 89.03 | 2023-12-06 00:00:00 | 88.35 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2023-12-29 00:00:00 | 90.78 | 2024-01-10 00:00:00 | 87.63 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest1 | 2024-02-05 00:00:00 | 89.63 | 2024-02-09 00:00:00 | 86.07 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest1 | 2024-04-01 00:00:00 | 87.33 | 2024-04-16 00:00:00 | 88.10 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest1 | 2024-04-26 00:00:00 | 92.50 | 2024-04-30 00:00:00 | 96.86 | PARTIAL | 0.50 | 4.71% |
