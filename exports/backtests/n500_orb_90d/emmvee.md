# Emmvee Photovoltaic Power Ltd. (EMMVEE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 269.25
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.09% / 0.28%
- **Sum % (uncompounded):** 0.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.09% | 0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.09% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.09% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 197.83 | 196.99 | 0.00 | ORB-long ORB[195.46,197.80] vol=1.5x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-03-10 09:45:00 | 196.85 | 197.04 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 266.25 | 264.38 | 0.00 | ORB-long ORB[261.22,265.04] vol=1.9x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 269.01 | 266.38 | 0.00 | T1 1.5R @ 269.01 |
| Target hit | 2026-04-21 11:00:00 | 267.00 | 268.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 264.00 | 262.52 | 0.00 | ORB-long ORB[260.77,263.90] vol=1.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 262.79 | 262.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-10 09:40:00 | 197.83 | 2026-03-10 09:45:00 | 196.85 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-21 09:30:00 | 266.25 | 2026-04-21 09:35:00 | 269.01 | PARTIAL | 0.50 | 1.04% |
| BUY | retest1 | 2026-04-21 09:30:00 | 266.25 | 2026-04-21 11:00:00 | 267.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2026-04-22 09:40:00 | 264.00 | 2026-04-22 09:50:00 | 262.79 | STOP_HIT | 1.00 | -0.46% |
