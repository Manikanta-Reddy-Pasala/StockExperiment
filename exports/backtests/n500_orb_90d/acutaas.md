# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2748.00
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
- **Avg / median % per leg:** 0.91% / -0.36%
- **Sum % (uncompounded):** 4.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.91% | 4.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.91% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.91% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 10:00:00 | 2114.10 | 2088.72 | 0.00 | ORB-long ORB[2062.00,2091.00] vol=2.1x ATR=9.90 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 2104.20 | 2094.12 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:30:00 | 2205.80 | 2190.01 | 0.00 | ORB-long ORB[2170.00,2198.70] vol=2.1x ATR=14.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:40:00 | 2226.93 | 2201.13 | 0.00 | T1 1.5R @ 2226.93 |
| Target hit | 2026-03-17 15:20:00 | 2313.50 | 2278.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2311.80 | 2298.23 | 0.00 | ORB-long ORB[2283.00,2306.00] vol=1.7x ATR=10.54 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 2301.26 | 2303.72 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 2755.00 | 2735.90 | 0.00 | ORB-long ORB[2700.00,2739.00] vol=3.3x ATR=9.81 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 2745.19 | 2736.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-16 10:00:00 | 2114.10 | 2026-03-16 10:15:00 | 2104.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-17 09:30:00 | 2205.80 | 2026-03-17 09:40:00 | 2226.93 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2026-03-17 09:30:00 | 2205.80 | 2026-03-17 15:20:00 | 2313.50 | TARGET_HIT | 0.50 | 4.88% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2311.80 | 2026-03-18 09:55:00 | 2301.26 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-05-08 09:30:00 | 2755.00 | 2026-05-08 09:35:00 | 2745.19 | STOP_HIT | 1.00 | -0.36% |
