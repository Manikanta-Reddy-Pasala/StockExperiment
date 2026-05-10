# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 145.00
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
- **Avg / median % per leg:** -0.13% / -0.28%
- **Sum % (uncompounded):** -0.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.09% | 0.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.09% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.13% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 133.46 | 133.88 | 0.00 | ORB-short ORB[133.62,135.00] vol=1.7x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 133.84 | 133.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 134.64 | 133.43 | 0.00 | ORB-long ORB[132.80,134.60] vol=4.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:35:00 | 135.32 | 133.68 | 0.00 | T1 1.5R @ 135.32 |
| Target hit | 2026-04-23 12:00:00 | 134.83 | 135.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:55:00 | 134.94 | 135.94 | 0.00 | ORB-short ORB[135.44,137.22] vol=1.5x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 135.80 | 135.46 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 143.98 | 142.96 | 0.00 | ORB-long ORB[142.09,143.40] vol=2.4x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-05-08 10:00:00 | 143.43 | 143.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-25 11:10:00 | 133.46 | 2026-02-25 11:30:00 | 133.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-23 11:15:00 | 134.64 | 2026-04-23 11:35:00 | 135.32 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-23 11:15:00 | 134.64 | 2026-04-23 12:00:00 | 134.83 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-04-30 09:55:00 | 134.94 | 2026-04-30 12:15:00 | 135.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2026-05-08 09:45:00 | 143.98 | 2026-05-08 10:00:00 | 143.43 | STOP_HIT | 1.00 | -0.38% |
