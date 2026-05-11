# Sammaan Capital Ltd. (SAMMAANCAP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 145.79
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 3.53% / 8.75%
- **Sum % (uncompounded):** 21.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.53% | 21.2% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.53% | 21.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.53% | 21.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 00:00:00 | 154.15 | 113.13 | 137.72 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=7.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 00:00:00 | 169.25 | 113.71 | 140.90 | T1 booked 50% @ 169.25 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 154.15 | 119.41 | 158.29 | SL hit (bars_held=12) |

### Cycle 2 — BUY (started 2023-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 00:00:00 | 154.05 | 130.56 | 149.59 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=6.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 167.53 | 132.26 | 153.13 | T1 booked 50% @ 167.53 |
| Target hit | 2023-12-20 00:00:00 | 180.90 | 142.00 | 181.91 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 191.65 | 152.74 | 183.87 | Stage2 pullback-breakout RSI=56 vol=2.9x ATR=9.68 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 177.13 | 155.68 | 184.95 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-02-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 00:00:00 | 207.00 | 158.66 | 188.71 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=10.00 |
| Stop hit — per-position SL triggered | 2024-02-28 00:00:00 | 192.01 | 159.71 | 189.97 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-25 00:00:00 | 154.15 | 2023-08-28 00:00:00 | 169.25 | PARTIAL | 0.50 | 9.80% |
| BUY | retest1 | 2023-08-25 00:00:00 | 154.15 | 2023-09-12 00:00:00 | 154.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 00:00:00 | 154.05 | 2023-11-16 00:00:00 | 167.53 | PARTIAL | 0.50 | 8.75% |
| BUY | retest1 | 2023-11-07 00:00:00 | 154.05 | 2023-12-20 00:00:00 | 180.90 | TARGET_HIT | 0.50 | 17.43% |
| BUY | retest1 | 2024-01-30 00:00:00 | 191.65 | 2024-02-12 00:00:00 | 177.13 | STOP_HIT | 1.00 | -7.57% |
| BUY | retest1 | 2024-02-23 00:00:00 | 207.00 | 2024-02-28 00:00:00 | 192.01 | STOP_HIT | 1.00 | -7.24% |
