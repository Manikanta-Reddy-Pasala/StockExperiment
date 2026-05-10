# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 866.00
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
- **Avg / median % per leg:** 0.29% / -0.41%
- **Sum % (uncompounded):** 1.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.46% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.79% | 2.4% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.79% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.29% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 562.45 | 555.95 | 0.00 | ORB-long ORB[550.00,556.00] vol=1.5x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 559.67 | 556.83 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 537.70 | 540.64 | 0.00 | ORB-short ORB[538.00,545.30] vol=2.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 535.81 | 539.94 | 0.00 | T1 1.5R @ 535.81 |
| Target hit | 2026-03-11 15:20:00 | 524.65 | 528.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 542.15 | 538.29 | 0.00 | ORB-long ORB[535.65,541.75] vol=1.5x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 539.82 | 539.87 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 543.60 | 547.89 | 0.00 | ORB-short ORB[546.20,554.10] vol=2.1x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-03-18 10:25:00 | 545.84 | 547.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-25 10:00:00 | 562.45 | 2026-02-25 10:05:00 | 559.67 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-11 10:35:00 | 537.70 | 2026-03-11 11:20:00 | 535.81 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-03-11 10:35:00 | 537.70 | 2026-03-11 15:20:00 | 524.65 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2026-03-17 09:45:00 | 542.15 | 2026-03-17 11:20:00 | 539.82 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-18 10:00:00 | 543.60 | 2026-03-18 10:25:00 | 545.84 | STOP_HIT | 1.00 | -0.41% |
