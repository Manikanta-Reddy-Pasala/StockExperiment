# Federal Bank Ltd. (FEDERALBNK)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 294.60
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.65% / -0.66%
- **Sum % (uncompounded):** 3.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.65% | 3.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.65% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.65% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 181.46 | 156.31 | 173.70 | Stage2 pullback-breakout RSI=67 vol=2.3x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 00:00:00 | 190.47 | 158.12 | 180.12 | T1 booked 50% @ 190.47 |
| Target hit | 2024-08-05 00:00:00 | 192.85 | 163.99 | 194.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 00:00:00 | 192.75 | 172.63 | 189.11 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 186.41 | 174.01 | 190.90 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-10-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 00:00:00 | 196.74 | 174.73 | 190.20 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=5.29 |
| Stop hit — per-position SL triggered | 2024-10-23 00:00:00 | 188.80 | 175.98 | 191.49 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 200.70 | 176.54 | 191.21 | Stage2 pullback-breakout RSI=62 vol=4.7x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 199.38 | 179.48 | 200.25 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 181.46 | 2024-07-11 00:00:00 | 190.47 | PARTIAL | 0.50 | 4.97% |
| BUY | retest1 | 2024-07-03 00:00:00 | 181.46 | 2024-08-05 00:00:00 | 192.85 | TARGET_HIT | 0.50 | 6.28% |
| BUY | retest1 | 2024-09-25 00:00:00 | 192.75 | 2024-10-07 00:00:00 | 186.41 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2024-10-14 00:00:00 | 196.74 | 2024-10-23 00:00:00 | 188.80 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest1 | 2024-10-29 00:00:00 | 200.70 | 2024-11-13 00:00:00 | 199.38 | STOP_HIT | 1.00 | -0.66% |
