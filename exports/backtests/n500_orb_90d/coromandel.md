# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1928.90
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
- **Avg / median % per leg:** 0.20% / -0.29%
- **Sum % (uncompounded):** 0.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.96% | 1.9% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 0.96% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.31% | -0.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.31% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.20% | 1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2262.60 | 2265.77 | 0.00 | ORB-short ORB[2271.90,2288.90] vol=1.9x ATR=7.72 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 2270.32 | 2265.38 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 2327.10 | 2313.95 | 0.00 | ORB-long ORB[2280.00,2313.60] vol=1.8x ATR=7.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:00:00 | 2338.52 | 2317.90 | 0.00 | T1 1.5R @ 2338.52 |
| Target hit | 2026-02-17 15:20:00 | 2360.30 | 2359.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 2255.10 | 2271.57 | 0.00 | ORB-short ORB[2276.40,2293.00] vol=1.5x ATR=6.54 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 2261.64 | 2267.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 2016.90 | 2043.65 | 0.00 | ORB-short ORB[2046.10,2064.50] vol=3.1x ATR=5.95 |
| Stop hit — per-position SL triggered | 2026-04-22 11:30:00 | 2022.85 | 2041.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:00:00 | 2262.60 | 2026-02-10 11:25:00 | 2270.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 09:55:00 | 2327.10 | 2026-02-17 10:00:00 | 2338.52 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-17 09:55:00 | 2327.10 | 2026-02-17 15:20:00 | 2360.30 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-02-23 10:50:00 | 2255.10 | 2026-02-23 11:10:00 | 2261.64 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-22 11:15:00 | 2016.90 | 2026-04-22 11:30:00 | 2022.85 | STOP_HIT | 1.00 | -0.30% |
