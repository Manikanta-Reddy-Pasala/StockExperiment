# Urban Company Ltd. (URBANCO)

## Backtest Summary

- **Window:** 2026-03-09 09:15:00 → 2026-05-08 15:25:00 (3000 bars)
- **Last close:** 137.80
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 0.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.30% | 0.9% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.30% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.09% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 109.81 | 109.09 | 0.00 | ORB-long ORB[108.08,109.44] vol=1.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 09:50:00 | 110.72 | 109.88 | 0.00 | T1 1.5R @ 110.72 |
| Target hit | 2026-03-10 10:05:00 | 110.32 | 110.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 123.25 | 124.23 | 0.00 | ORB-short ORB[123.65,125.40] vol=1.7x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-04-09 10:05:00 | 123.76 | 124.12 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 135.33 | 136.77 | 0.00 | ORB-short ORB[136.02,137.99] vol=1.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-15 11:30:00 | 135.89 | 136.66 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 136.29 | 134.94 | 0.00 | ORB-long ORB[134.17,135.69] vol=7.2x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 135.78 | 135.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-05-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:55:00 | 143.85 | 145.56 | 0.00 | ORB-short ORB[145.31,146.98] vol=2.0x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:05:00 | 143.05 | 145.07 | 0.00 | T1 1.5R @ 143.05 |
| Stop hit — per-position SL triggered | 2026-05-08 12:20:00 | 143.85 | 144.94 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-10 09:30:00 | 109.81 | 2026-03-10 09:50:00 | 110.72 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-03-10 09:30:00 | 109.81 | 2026-03-10 10:05:00 | 110.32 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-09 09:50:00 | 123.25 | 2026-04-09 10:05:00 | 123.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-15 11:00:00 | 135.33 | 2026-04-15 11:30:00 | 135.89 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-21 11:00:00 | 136.29 | 2026-04-21 11:05:00 | 135.78 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-08 10:55:00 | 143.85 | 2026-05-08 12:05:00 | 143.05 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-08 10:55:00 | 143.85 | 2026-05-08 12:20:00 | 143.85 | STOP_HIT | 0.50 | 0.00% |
