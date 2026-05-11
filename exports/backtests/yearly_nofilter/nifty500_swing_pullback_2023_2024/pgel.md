# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 521.25
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 3.39% / 1.64%
- **Sum % (uncompounded):** 10.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.39% | 10.2% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.39% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 3.39% | 10.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 00:00:00 | 161.14 | 135.22 | 153.46 | Stage2 pullback-breakout RSI=64 vol=5.5x ATR=5.86 |
| Stop hit — per-position SL triggered | 2023-08-22 00:00:00 | 162.59 | 137.88 | 159.35 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 00:00:00 | 187.58 | 149.40 | 178.70 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 00:00:00 | 201.91 | 152.49 | 187.15 | T1 booked 50% @ 201.91 |
| Target hit | 2023-10-25 00:00:00 | 190.66 | 154.41 | 191.71 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-07 00:00:00 | 161.14 | 2023-08-22 00:00:00 | 162.59 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2023-10-09 00:00:00 | 187.58 | 2023-10-18 00:00:00 | 201.91 | PARTIAL | 0.50 | 7.64% |
| BUY | retest1 | 2023-10-09 00:00:00 | 187.58 | 2023-10-25 00:00:00 | 190.66 | TARGET_HIT | 0.50 | 1.64% |
