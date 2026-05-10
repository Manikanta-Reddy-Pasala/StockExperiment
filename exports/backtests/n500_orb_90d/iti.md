# ITI Ltd. (ITI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 300.30
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 0.08% / 0.26%
- **Sum % (uncompounded):** 0.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.17% | -0.8% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.17% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.32% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.08% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 292.20 | 294.27 | 0.00 | ORB-short ORB[294.10,298.00] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-11 11:50:00 | 292.91 | 293.92 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 284.25 | 285.82 | 0.00 | ORB-short ORB[285.30,288.00] vol=2.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:20:00 | 283.24 | 285.00 | 0.00 | T1 1.5R @ 283.24 |
| Target hit | 2026-02-19 15:20:00 | 281.00 | 283.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 278.95 | 280.82 | 0.00 | ORB-short ORB[280.90,282.75] vol=3.7x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 15:00:00 | 277.98 | 279.59 | 0.00 | T1 1.5R @ 277.98 |
| Stop hit — per-position SL triggered | 2026-02-23 15:05:00 | 278.95 | 279.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 261.80 | 257.48 | 0.00 | ORB-long ORB[255.00,257.60] vol=5.0x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-03-05 09:55:00 | 260.18 | 257.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:25:00 | 303.81 | 299.33 | 0.00 | ORB-long ORB[291.28,294.90] vol=12.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 301.86 | 299.68 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 295.28 | 293.35 | 0.00 | ORB-long ORB[291.00,294.30] vol=2.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 296.99 | 296.47 | 0.00 | T1 1.5R @ 296.99 |
| Target hit | 2026-04-21 10:25:00 | 296.05 | 296.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 303.04 | 300.92 | 0.00 | ORB-long ORB[298.99,302.35] vol=2.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-27 09:50:00 | 301.80 | 301.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:00:00 | 292.20 | 2026-02-11 11:50:00 | 292.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-19 09:30:00 | 284.25 | 2026-02-19 10:20:00 | 283.24 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-19 09:30:00 | 284.25 | 2026-02-19 15:20:00 | 281.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2026-02-23 11:00:00 | 278.95 | 2026-02-23 15:00:00 | 277.98 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-23 11:00:00 | 278.95 | 2026-02-23 15:05:00 | 278.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 09:50:00 | 261.80 | 2026-03-05 09:55:00 | 260.18 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2026-04-17 10:25:00 | 303.81 | 2026-04-17 10:30:00 | 301.86 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2026-04-21 09:40:00 | 295.28 | 2026-04-21 10:05:00 | 296.99 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-21 09:40:00 | 295.28 | 2026-04-21 10:25:00 | 296.05 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-27 09:45:00 | 303.04 | 2026-04-27 09:50:00 | 301.80 | STOP_HIT | 1.00 | -0.41% |
