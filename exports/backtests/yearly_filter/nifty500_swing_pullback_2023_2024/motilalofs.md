# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 863.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 3
- **Avg / median % per leg:** 7.74% / 6.29%
- **Sum % (uncompounded):** 54.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 7.74% | 54.2% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 7.74% | 54.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 2 | 3 | 7.74% | 54.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 00:00:00 | 195.60 | 170.75 | 183.58 | Stage2 pullback-breakout RSI=68 vol=2.9x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 00:00:00 | 206.45 | 171.13 | 186.00 | T1 booked 50% @ 206.45 |
| Target hit | 2023-08-28 00:00:00 | 216.35 | 180.51 | 217.65 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 229.31 | 189.10 | 219.58 | Stage2 pullback-breakout RSI=61 vol=1.9x ATR=7.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 00:00:00 | 243.72 | 192.01 | 226.42 | T1 booked 50% @ 243.72 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 229.31 | 195.67 | 235.94 | SL hit (bars_held=14) |

### Cycle 3 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 259.51 | 198.08 | 240.03 | Stage2 pullback-breakout RSI=64 vol=2.6x ATR=10.86 |
| Stop hit — per-position SL triggered | 2023-11-02 00:00:00 | 243.23 | 199.09 | 241.70 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 321.91 | 233.48 | 308.00 | Stage2 pullback-breakout RSI=61 vol=2.8x ATR=12.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 00:00:00 | 346.01 | 236.46 | 314.95 | T1 booked 50% @ 346.01 |
| Target hit | 2024-02-21 00:00:00 | 420.11 | 285.65 | 433.43 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 00:00:00 | 195.60 | 2023-07-26 00:00:00 | 206.45 | PARTIAL | 0.50 | 5.55% |
| BUY | retest1 | 2023-07-25 00:00:00 | 195.60 | 2023-08-28 00:00:00 | 216.35 | TARGET_HIT | 0.50 | 10.61% |
| BUY | retest1 | 2023-10-03 00:00:00 | 229.31 | 2023-10-12 00:00:00 | 243.72 | PARTIAL | 0.50 | 6.29% |
| BUY | retest1 | 2023-10-03 00:00:00 | 229.31 | 2023-10-23 00:00:00 | 229.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 00:00:00 | 259.51 | 2023-11-02 00:00:00 | 243.23 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest1 | 2024-01-04 00:00:00 | 321.91 | 2024-01-09 00:00:00 | 346.01 | PARTIAL | 0.50 | 7.49% |
| BUY | retest1 | 2024-01-04 00:00:00 | 321.91 | 2024-02-21 00:00:00 | 420.11 | TARGET_HIT | 0.50 | 30.51% |
