# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 339.60
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
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 0.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.35% | 0.7% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.35% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.08% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 264.20 | 261.68 | 0.00 | ORB-long ORB[259.26,262.75] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:35:00 | 266.05 | 263.40 | 0.00 | T1 1.5R @ 266.05 |
| Stop hit — per-position SL triggered | 2026-02-20 10:30:00 | 264.20 | 265.10 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 233.10 | 235.29 | 0.00 | ORB-short ORB[233.20,236.45] vol=2.8x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:20:00 | 231.50 | 233.94 | 0.00 | T1 1.5R @ 231.50 |
| Stop hit — per-position SL triggered | 2026-03-17 13:35:00 | 233.10 | 233.11 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 252.00 | 254.57 | 0.00 | ORB-short ORB[253.55,257.16] vol=1.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 253.12 | 254.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 289.82 | 293.50 | 0.00 | ORB-short ORB[292.46,296.71] vol=1.6x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 291.23 | 293.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 09:30:00 | 264.20 | 2026-02-20 09:35:00 | 266.05 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-20 09:30:00 | 264.20 | 2026-02-20 10:30:00 | 264.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 11:15:00 | 233.10 | 2026-03-17 12:20:00 | 231.50 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-17 11:15:00 | 233.10 | 2026-03-17 13:35:00 | 233.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:45:00 | 252.00 | 2026-04-16 09:50:00 | 253.12 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-24 09:35:00 | 289.82 | 2026-04-24 09:40:00 | 291.23 | STOP_HIT | 1.00 | -0.48% |
