# Rail Vikas Nigam Ltd. (RVNL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 305.15
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 4
- **Avg / median % per leg:** 0.64% / 0.60%
- **Sum % (uncompounded):** 5.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.32% | 1.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.32% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.96% | 3.8% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.96% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.64% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:50:00 | 256.65 | 258.43 | 0.00 | ORB-short ORB[257.45,260.75] vol=1.6x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 254.83 | 257.67 | 0.00 | T1 1.5R @ 254.83 |
| Target hit | 2026-03-23 15:20:00 | 250.10 | 253.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 295.00 | 291.40 | 0.00 | ORB-long ORB[290.00,294.16] vol=3.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:15:00 | 297.03 | 292.53 | 0.00 | T1 1.5R @ 297.03 |
| Stop hit — per-position SL triggered | 2026-04-16 11:35:00 | 295.00 | 293.14 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 305.50 | 307.67 | 0.00 | ORB-short ORB[306.43,309.88] vol=2.2x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:45:00 | 303.71 | 306.82 | 0.00 | T1 1.5R @ 303.71 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 305.50 | 306.58 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 300.20 | 298.05 | 0.00 | ORB-long ORB[295.45,297.80] vol=3.2x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:05:00 | 302.01 | 299.17 | 0.00 | T1 1.5R @ 302.01 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 300.20 | 299.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-23 09:50:00 | 256.65 | 2026-03-23 10:15:00 | 254.83 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-23 09:50:00 | 256.65 | 2026-03-23 15:20:00 | 250.10 | TARGET_HIT | 0.50 | 2.55% |
| BUY | retest1 | 2026-04-16 11:00:00 | 295.00 | 2026-04-16 11:15:00 | 297.03 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-16 11:00:00 | 295.00 | 2026-04-16 11:35:00 | 295.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 305.50 | 2026-04-24 09:45:00 | 303.71 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-24 09:30:00 | 305.50 | 2026-04-24 10:00:00 | 305.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:35:00 | 300.20 | 2026-05-05 10:05:00 | 302.01 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-05-05 09:35:00 | 300.20 | 2026-05-05 10:25:00 | 300.20 | STOP_HIT | 0.50 | 0.00% |
