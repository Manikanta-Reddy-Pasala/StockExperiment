# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 314.85
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 5
- **Avg / median % per leg:** 0.32% / 0.00%
- **Sum % (uncompounded):** 4.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.12% | -0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.12% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.60% | 4.8% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.60% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 6 | 46.2% | 1 | 7 | 5 | 0.32% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 391.45 | 384.86 | 0.00 | ORB-long ORB[379.30,383.80] vol=3.4x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 389.12 | 386.50 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 393.00 | 393.94 | 0.00 | ORB-short ORB[394.35,399.00] vol=2.6x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:45:00 | 391.01 | 393.68 | 0.00 | T1 1.5R @ 391.01 |
| Stop hit — per-position SL triggered | 2026-02-18 13:00:00 | 393.00 | 393.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 282.50 | 280.20 | 0.00 | ORB-long ORB[277.25,281.40] vol=2.3x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:45:00 | 284.76 | 280.68 | 0.00 | T1 1.5R @ 284.76 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 282.50 | 281.07 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 277.00 | 279.66 | 0.00 | ORB-short ORB[279.05,282.85] vol=1.9x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:10:00 | 275.35 | 279.04 | 0.00 | T1 1.5R @ 275.35 |
| Target hit | 2026-03-19 15:20:00 | 270.85 | 273.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:40:00 | 269.50 | 271.92 | 0.00 | ORB-short ORB[270.70,274.65] vol=5.4x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:10:00 | 267.42 | 271.06 | 0.00 | T1 1.5R @ 267.42 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 269.50 | 270.76 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 298.25 | 300.00 | 0.00 | ORB-short ORB[298.50,303.00] vol=1.7x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:05:00 | 296.12 | 299.11 | 0.00 | T1 1.5R @ 296.12 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 298.25 | 298.49 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 300.00 | 298.17 | 0.00 | ORB-long ORB[295.75,299.10] vol=1.7x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-29 09:40:00 | 298.76 | 298.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 293.70 | 292.09 | 0.00 | ORB-long ORB[290.60,292.90] vol=2.4x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 292.54 | 292.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 09:45:00 | 391.45 | 2026-02-13 10:15:00 | 389.12 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2026-02-18 11:05:00 | 393.00 | 2026-02-18 11:45:00 | 391.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-18 11:05:00 | 393.00 | 2026-02-18 13:00:00 | 393.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-16 09:40:00 | 282.50 | 2026-03-16 09:45:00 | 284.76 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-03-16 09:40:00 | 282.50 | 2026-03-16 09:55:00 | 282.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 11:00:00 | 277.00 | 2026-03-19 11:10:00 | 275.35 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-19 11:00:00 | 277.00 | 2026-03-19 15:20:00 | 270.85 | TARGET_HIT | 0.50 | 2.22% |
| SELL | retest1 | 2026-03-20 10:40:00 | 269.50 | 2026-03-20 11:10:00 | 267.42 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-20 10:40:00 | 269.50 | 2026-03-20 11:20:00 | 269.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 09:45:00 | 298.25 | 2026-04-10 10:05:00 | 296.12 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-04-10 09:45:00 | 298.25 | 2026-04-10 10:50:00 | 298.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:35:00 | 300.00 | 2026-04-29 09:40:00 | 298.76 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-06 09:30:00 | 293.70 | 2026-05-06 09:35:00 | 292.54 | STOP_HIT | 1.00 | -0.39% |
