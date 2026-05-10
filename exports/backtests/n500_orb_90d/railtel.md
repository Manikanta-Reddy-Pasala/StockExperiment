# Railtel Corporation Of India Ltd. (RAILTEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 343.35
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
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 0.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.05% | -0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.05% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.14% | 0.4% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.14% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.03% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 319.00 | 321.63 | 0.00 | ORB-short ORB[321.10,325.60] vol=2.9x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 319.83 | 321.46 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 280.35 | 282.82 | 0.00 | ORB-short ORB[282.75,285.80] vol=2.3x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:45:00 | 278.47 | 281.72 | 0.00 | T1 1.5R @ 278.47 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 280.35 | 281.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 333.30 | 328.89 | 0.00 | ORB-long ORB[325.01,329.35] vol=6.9x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-04-22 10:50:00 | 331.90 | 329.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 322.00 | 320.50 | 0.00 | ORB-long ORB[317.05,321.49] vol=2.6x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:45:00 | 323.92 | 321.74 | 0.00 | T1 1.5R @ 323.92 |
| Target hit | 2026-04-27 14:30:00 | 322.55 | 322.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 339.90 | 336.11 | 0.00 | ORB-long ORB[332.05,335.40] vol=2.4x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 337.99 | 337.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-25 11:05:00 | 319.00 | 2026-02-25 11:30:00 | 319.83 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-13 09:50:00 | 280.35 | 2026-03-13 10:45:00 | 278.47 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-03-13 09:50:00 | 280.35 | 2026-03-13 10:50:00 | 280.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 333.30 | 2026-04-22 10:50:00 | 331.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-27 09:30:00 | 322.00 | 2026-04-27 10:45:00 | 323.92 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:30:00 | 322.00 | 2026-04-27 14:30:00 | 322.55 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-05-05 09:35:00 | 339.90 | 2026-05-05 09:45:00 | 337.99 | STOP_HIT | 1.00 | -0.56% |
