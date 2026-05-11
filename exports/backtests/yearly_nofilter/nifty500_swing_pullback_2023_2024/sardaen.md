# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 583.60
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 4.48% / 9.04%
- **Sum % (uncompounded):** 26.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 2 | 2 | 4.48% | 26.9% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 2 | 2 | 4.48% | 26.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 2 | 2 | 4.48% | 26.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 00:00:00 | 199.10 | 129.60 | 186.38 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=9.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 00:00:00 | 217.10 | 136.46 | 200.57 | T1 booked 50% @ 217.10 |
| Target hit | 2023-09-12 00:00:00 | 217.75 | 144.09 | 219.17 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 228.65 | 159.41 | 219.09 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=9.47 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 214.44 | 161.67 | 218.11 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 231.10 | 169.57 | 215.39 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=11.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 253.28 | 176.77 | 231.76 | T1 booked 50% @ 253.28 |
| Target hit | 2023-12-20 00:00:00 | 240.75 | 185.55 | 247.29 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 00:00:00 | 230.10 | 205.73 | 212.74 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=10.16 |
| Stop hit — per-position SL triggered | 2024-04-26 00:00:00 | 232.15 | 207.85 | 222.92 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-18 00:00:00 | 199.10 | 2023-08-31 00:00:00 | 217.10 | PARTIAL | 0.50 | 9.04% |
| BUY | retest1 | 2023-08-18 00:00:00 | 199.10 | 2023-09-12 00:00:00 | 217.75 | TARGET_HIT | 0.50 | 9.37% |
| BUY | retest1 | 2023-10-17 00:00:00 | 228.65 | 2023-10-23 00:00:00 | 214.44 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2023-11-16 00:00:00 | 231.10 | 2023-12-04 00:00:00 | 253.28 | PARTIAL | 0.50 | 9.60% |
| BUY | retest1 | 2023-11-16 00:00:00 | 231.10 | 2023-12-20 00:00:00 | 240.75 | TARGET_HIT | 0.50 | 4.18% |
| BUY | retest1 | 2024-04-10 00:00:00 | 230.10 | 2024-04-26 00:00:00 | 232.15 | STOP_HIT | 1.00 | 0.89% |
