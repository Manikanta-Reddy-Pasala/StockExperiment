# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 14571.00
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
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 2.55% / 1.63%
- **Sum % (uncompounded):** 12.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 3 | 1 | 2.55% | 12.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 3 | 1 | 2.55% | 12.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 3 | 1 | 2.55% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 05:30:00 | 13820.20 | 11210.76 | 13176.15 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=453.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 05:30:00 | 14726.60 | 11246.56 | 13331.55 | T1 booked 50% @ 14726.60 |
| Target hit | 2023-10-04 05:30:00 | 15162.80 | 12052.91 | 15189.84 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 05:30:00 | 16294.80 | 12790.04 | 15713.20 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=506.28 |
| Stop hit — per-position SL triggered | 2023-11-21 05:30:00 | 16297.80 | 13137.61 | 16166.26 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 05:30:00 | 16006.90 | 13738.92 | 15712.90 | Stage2 pullback-breakout RSI=55 vol=3.0x ATR=487.60 |
| Stop hit — per-position SL triggered | 2024-01-12 05:30:00 | 16267.80 | 13966.36 | 15973.42 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 05:30:00 | 15505.35 | 14387.75 | 14630.38 | Stage2 pullback-breakout RSI=61 vol=6.0x ATR=533.81 |
| Stop hit — per-position SL triggered | 2024-03-27 05:30:00 | 14704.63 | 14450.78 | 14947.48 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 05:30:00 | 13820.20 | 2023-09-01 05:30:00 | 14726.60 | PARTIAL | 0.50 | 6.56% |
| BUY | retest1 | 2023-08-31 05:30:00 | 13820.20 | 2023-10-04 05:30:00 | 15162.80 | TARGET_HIT | 0.50 | 9.71% |
| BUY | retest1 | 2023-11-07 05:30:00 | 16294.80 | 2023-11-21 05:30:00 | 16297.80 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2023-12-29 05:30:00 | 16006.90 | 2024-01-12 05:30:00 | 16267.80 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest1 | 2024-03-15 05:30:00 | 15505.35 | 2024-03-27 05:30:00 | 14704.63 | STOP_HIT | 1.00 | -5.16% |
