# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4118.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 0.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.12% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.37% | 0.7% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.37% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.05% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 2909.40 | 2886.36 | 0.00 | ORB-long ORB[2868.00,2901.90] vol=2.2x ATR=12.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:40:00 | 2928.09 | 2898.80 | 0.00 | T1 1.5R @ 2928.09 |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 2909.40 | 2912.15 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 3455.00 | 3429.70 | 0.00 | ORB-long ORB[3390.00,3440.00] vol=2.4x ATR=18.11 |
| Stop hit — per-position SL triggered | 2026-04-10 09:40:00 | 3436.89 | 3440.55 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 3494.90 | 3448.49 | 0.00 | ORB-long ORB[3400.00,3449.80] vol=3.5x ATR=20.33 |
| Stop hit — per-position SL triggered | 2026-04-15 10:00:00 | 3474.57 | 3460.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 3482.50 | 3508.01 | 0.00 | ORB-short ORB[3485.70,3534.90] vol=2.4x ATR=17.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 3456.41 | 3500.20 | 0.00 | T1 1.5R @ 3456.41 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 3482.50 | 3482.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 10:05:00 | 2909.40 | 2026-02-19 10:40:00 | 2928.09 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-19 10:05:00 | 2909.40 | 2026-02-19 11:15:00 | 2909.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 3455.00 | 2026-04-10 09:40:00 | 3436.89 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-15 09:45:00 | 3494.90 | 2026-04-15 10:00:00 | 3474.57 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-04-16 09:30:00 | 3482.50 | 2026-04-16 09:40:00 | 3456.41 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-04-16 09:30:00 | 3482.50 | 2026-04-16 10:25:00 | 3482.50 | STOP_HIT | 0.50 | 0.00% |
