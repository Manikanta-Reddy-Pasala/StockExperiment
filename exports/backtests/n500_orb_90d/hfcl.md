# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 139.75
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
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.14% / -0.35%
- **Sum % (uncompounded):** -0.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.41% | -0.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.41% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.14% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 72.21 | 71.30 | 0.00 | ORB-long ORB[70.81,71.39] vol=2.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 71.84 | 71.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:40:00 | 70.15 | 70.76 | 0.00 | ORB-short ORB[70.54,71.53] vol=1.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-12 11:10:00 | 70.40 | 70.72 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 68.99 | 69.46 | 0.00 | ORB-short ORB[69.22,70.15] vol=2.4x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 69.31 | 69.40 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:30:00 | 72.95 | 72.26 | 0.00 | ORB-long ORB[71.51,72.34] vol=2.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:35:00 | 73.40 | 72.50 | 0.00 | T1 1.5R @ 73.40 |
| Stop hit — per-position SL triggered | 2026-03-18 10:40:00 | 72.95 | 72.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 72.21 | 2026-02-09 11:30:00 | 71.84 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-02-12 10:40:00 | 70.15 | 2026-02-12 11:10:00 | 70.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-24 09:30:00 | 68.99 | 2026-02-24 09:35:00 | 69.31 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-03-18 10:30:00 | 72.95 | 2026-03-18 10:35:00 | 73.40 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-03-18 10:30:00 | 72.95 | 2026-03-18 10:40:00 | 72.95 | STOP_HIT | 0.50 | 0.00% |
